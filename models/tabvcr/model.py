from typing import Dict, List, Any
import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.nn.modules import BatchNorm2d,BatchNorm1d
from utils.pytorch_misc import Flattener
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

# image backbone code from https://github.com/rowanz/r2c/blob/master/utils/detector.py
def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone

@Model.register("LSTMBatchNormBUANonTagGlobalFullNoFinalImage")
class LSTMBatchNormBUANonTagGlobalFullNoFinalImage(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 option_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(LSTMBatchNormBUANonTagGlobalFullNoFinalImage, self).__init__(vocab)
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.image_BN = BatchNorm1d(512)

        self.option_encoder = TimeDistributed(option_encoder)
        self.option_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.query_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.final_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.final_mlp_linear = torch.nn.Sequential(
            torch.nn.Linear(512,1)
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)
    # recevie redundent parameters for convinence        

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]
 
        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
           row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return span_rep, retrieved_feats


    def forward(self,
                det_features:torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """

        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        # objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        det_features = det_features[:,:max_len,:]
        # segms = segms[:, :max_len]

        obj_reps = det_features
        obj_reps = self.obj_downsample(obj_reps)

        # obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        # option part
        batch_size, num_options, padded_seq_len, _ = answers['bert'].shape
        options, option_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps)
        assert (options.shape == (batch_size, num_options, padded_seq_len, 1280))
        option_rep = self.option_encoder(options, answer_mask) # (batch_size, 4, seq_len, emb_len(512))
        option_rep = replace_masked_values(option_rep, answer_mask[...,None], 0)

        seq_real_length = torch.sum(answer_mask, dim=-1, dtype=torch.float) # (batch_size, 4)
        seq_real_length = seq_real_length.view(-1,1) # (batch_size * 4,1)

        option_rep = option_rep.sum(dim=2) # (batch_size, 4, emb_len(512))
        option_rep = option_rep.view(batch_size * num_options,512) # (batch_size * 4, emb_len(512))
        option_rep = option_rep.div(seq_real_length) # (batch_size * 4, emb_len(512))
        option_rep = self.option_BN(option_rep)
        option_rep = option_rep.view(batch_size, num_options, 512) # (batch_size, 4, emb_len(512))

        # query part
        batch_size, num_options, padded_seq_len, _ = question['bert'].shape
        query, query_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps)
        assert (query.shape == (batch_size, num_options, padded_seq_len, 1280))
        query_rep = self.option_encoder(query, question_mask) # (batch_size, 4, seq_len, emb_len(512))
        query_rep = replace_masked_values(query_rep, question_mask[...,None], 0)

        seq_real_length = torch.sum(question_mask, dim=-1, dtype=torch.float) # (batch_size, 4)
        seq_real_length = seq_real_length.view(-1,1) # (batch_size * 4,1)

        query_rep = query_rep.sum(dim=2) # (batch_size, 4, emb_len(512))
        query_rep = query_rep.view(batch_size * num_options,512) # (batch_size * 4, emb_len(512))
        query_rep = query_rep.div(seq_real_length) # (batch_size * 4, emb_len(512))
        query_rep = self.query_BN(query_rep)
        query_rep = query_rep.view(batch_size, num_options, 512) # (batch_size, 4, emb_len(512))

        # image part

        # assert (obj_reps[:,0,:].shape == (batch_size, 512))
        # images = obj_reps[:,0,:] #  the background i.e. whole image
        # images = self.image_BN(images)
        # images = images[:,None,:]
        # images = images.repeat(1,4,1) # (batch_size, 4, 512)
        # assert (images.shape == (batch_size, num_options,512))

        query_option_image_cat = torch.cat((option_rep,query_rep),-1)
        assert (query_option_image_cat.shape == (batch_size,num_options, 512*2))
        query_option_image_cat = self.final_mlp(query_option_image_cat)
        query_option_image_cat = query_option_image_cat.view(batch_size*num_options,512)
        query_option_image_cat = self.final_BN(query_option_image_cat)
        query_option_image_cat = query_option_image_cat.view(batch_size,num_options,512)
        logits = self.final_mlp_linear(query_option_image_cat)
        logits = logits.squeeze(2)
        class_probabilities = F.softmax(logits, dim=-1)
        output_dict = {"label_logits": logits, "label_probs": class_probabilities}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]

        # print ('one pass')
        return output_dict
    def get_metrics(self,reset=False):
        return {'accuracy': self._accuracy.get_metric(reset)}




