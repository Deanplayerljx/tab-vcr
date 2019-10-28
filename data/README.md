# Data

Obtain the dataset by visiting [visualcommonsense.com/download.html](https://visualcommonsense.com/download.html). 
 - Extract the images somewhere and added a symlink in this (`data`): `ln -s {path_to_image_folder}`
 - Put `train.jsonl`, `val.jsonl`, and `test.jsonl` in here (`data`).
 
You can also put the dataset somewhere else, you'll just need to update `config.py` (in the main directory) accordingly.

# Precomputed representations
Running TAB-VCR requires pre-computed representations in this folder. 

1. BERT representation provided by VCR dataset:
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_train.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_val.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_answer_test.h5`
    * `https://s3-us-west-2.amazonaws.com/ai2-rowanz/r2c/bert_da_rationale_test.h5`
    
  
2. Attribute and New Tag features generated using code in [Bottom Up Attention](https://github.com/peteanderson80/bottom-up-attention), released by paper [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering, Peter Anderson et al., 2018](https://arxiv.org/abs/1707.07998):
    * `https://vcr-bert.s3.amazonaws.com/bert_options/attribute_features_train.h5`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/attribute_features_val.h5`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/attribute_features_test.h5`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_train.h5`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_val.h5`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_test.h5`
  
3. (optional) Lightweight new tag features (bounding box info only). Needed only if you want to run new tag matching script in `similarity_match` folder on your own. 
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_light_weight_train.tsv`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_light_weight_val.tsv`
    * `https://vcr-bert.s3.amazonaws.com/bert_options/new_tag_features_light_weight_test_.tsv`
    