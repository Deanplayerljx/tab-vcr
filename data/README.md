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
    * `https://drive.google.com/open?id=19e4tM7yfWViKFe0zH2VpHKkIBwfGbZYL`
    * `https://drive.google.com/open?id=1xcfYOkyTJY0m89Xa8vbYCRdbFag26l5_`
    * `https://drive.google.com/open?id=1FLIsHK7-QEWPEjy0LfXSrQlmRqRziKCc`
    * `https://drive.google.com/open?id=1fSWPmZJ2pf_0uBntJRdjiI47a76FRRtF`
    * `https://drive.google.com/open?id=1Ersqwo1Qiqd-DLf8gWYoitu1GDQFeK82`
    * `https://drive.google.com/open?id=1EU64oGB1LHz_jjtawmCAMJeaYzftoks1`
    
3. (optional) Lightweight new tag features (bounding box info only). Needed only if you want to run new tag matching script in `similarity_match` folder on your own. 
    * `https://drive.google.com/a/illinois.edu/file/d/1e6Ei9dadYSAcznKt6VOixkY4VOKO2wgF/view?usp=sharing`
    
