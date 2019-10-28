# Dean's note
`vcr.py` is the dataloader that loads the train & val sets for full models

`my_vcr.py` is the dataloader that loads the train & val sets for models without object detections.

`my_vcr_no_image` is the dataloader that loads the train & val sets for no image models

`my_vcr_det.py` is the dataloader that loads the train & val sets for models with detections.

`my_vcr_vgg.py` is the dataloader that loads the train & val sets for vgg net, the only difference is that it resize the input image to 224 \* 224 instead of 786 \* 384 (the size original author use).

Please choose them approriately according to your model needs, and import it in `models/my_train.py`.