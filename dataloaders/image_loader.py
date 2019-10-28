"""
Dataloaders for VCR
"""
import json
import os

import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
import multiprocessing
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


class VCRImage(Dataset):
    def __init__(self, split,add_image_as_a_box=True):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        :param embs_to_load: Which precomputed embeddings to load.
        :param conditioned_answer_choice: If you're in test mode, the answer labels aren't provided, which could be
                                          a problem for the QA->R task. Pass in 'conditioned_answer_choice=i'
                                          to always condition on the i-th answer.
        """
        self.split = split
        self.add_image_as_a_box = add_image_as_a_box
        img_id_2_image_folder = {}
        img_id_2_meta_folder ={}
        with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(split)), 'r') as f:
            for s in f:
                item = json.loads(s)
                img_id_2_meta_folder[item['img_id']] = os.path.join(VCR_IMAGES_DIR, item['metadata_fn'])
                img_id_2_image_folder[item['img_id']] = os.path.join(VCR_IMAGES_DIR, item['img_fn'])
            self.img_ids = list(img_id_2_image_folder.keys())
        self.img_id_2_image_folder = img_id_2_image_folder
        self.img_id_2_meta_folder = img_id_2_meta_folder


        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))

        with open(os.path.join(os.path.dirname(VCR_ANNOTS_DIR), 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        # test = cls(split='test', **kwargs_copy)
        return train,val
        return train, val, test

    @property
    def is_train(self):
        return self.split == 'train'
    def __len__(self):
        print (len(self.img_ids))
        return len(self.img_ids)


    def __getitem__(self, index):
        # if self.split == 'test':
        #     raise ValueError("blind test mode not supported quite yet")
        img_id = self.img_ids[index]
        instance_dict = {}

        image = load_image(self.img_id_2_image_folder[img_id])
        image, window, img_scale, padding = resize_image(image, random_pad=False)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        # print (self.img_id_2_folder[img_id])
        with open(self.img_id_2_meta_folder[img_id], 'r') as f:
            metadata = json.load(f)

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[:, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]

        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))

        # if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            # import ipdb
            # ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        instance = Instance(instance_dict)
        if int(img_id.split('-')[-1]) == 53716:
            print ('find')
        return image, instance, int(img_id.split('-')[-1])


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    images, instances, img_ids = zip(*data)
    images = torch.stack(images, 0)
    batch = Batch(instances)
    td = batch.as_tensor_dict()

    td['img_ids'] = torch.LongTensor(list(img_ids))
    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
    td['images'] = images
    return td


class VCRImageLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, to_gpu=False),
            drop_last=False,
            pin_memory=False,
            **kwargs,
        )
        return loader

