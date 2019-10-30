# Models

This folder is for models reported in the paper. They broadly follow the allennlp configuration format. If you want TAB-VCR, you'll want to look at `tabvcr`.

## Model Training
- To train your own model for question answering, run:
```
python my_train.py -params {path_to_your_model_config} -folder
{path_to_save_model_checkpoints} -plot {plot name}
```
- To train your own model for answer justification, run:
```
python my_train.py -params {path_to_your_model_config} -folder
{path_to_save_model_checkpoints} -plot {plot name} -rationale
```

## Model Evaluation
- To evaluate the prediction results on question ansewring, run:
```
python eval_from_preds -preds {path_to_prediction_file} 
```

- To evaluate the prediction results on answer justification, run:
```
python eval_from_preds -preds {path_to_prediction_file} -rationale
```

- To evaluate the best model checkpoint for question answering, run (best
  checkpoint should be saved with name "best"):
```
python eval_best_checkpoint.py -params {path_to_your_model_config} -folder
{path_to_model_checkpoints}
```

- To evaluate the best model checkpoint for answer justification, run (best
  checkpoint should be saved with name "best"):
```
python eval_best_checkpoint.py -params {path_to_your_model_config} -folder
{path_to_model_checkpoints} -rationale
```

## Replicating validation results
We used 2 V100 GPUs with 12 workers (defined in my_train.py) for models reported in the paper.

Model checkpoints can be downloaded at `https://drive.google.com/a/illinois.edu/file/d/1B0PFR4FXIRuvesrLAANC8m0mZ8nhuP-H/view?usp=sharing`. Each folder contains the model checkpoint `best.th` and its predictions `valpreds.npy` on the validation set.

You can combine the validation predictions using
```
python eval_q2ar.py -answer_preds ../saves/tabvcr/valpreds.npy -rationale_preds ../saves/tabvcr/valpreds.npy
```

## Notes
- Once you defined your own model, don't forget to import it in \_\_init\_\_.py .





