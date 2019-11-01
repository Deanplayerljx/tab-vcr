# TAB-VCR: Tags and Attributes based VCR Baselines

This repository contains data and PyTorch code for the paper [TAB-VCR: Tags and Attributes based VCR Baselines](https://128.84.21.199/abs/1910.14671). 

Authors: Jingxiang Lin, [Unnat Jain](https://unnat.github.io), [Alexander G. Swching](http://www.alexander-schwing.de). 

Please check out the [project page](https://deanplayerljx.github.io/tabvcr) for more info.

# Setting up and using the repo

This repo is based on the [VCR dataset repo](https://github.com/rowanz/r2c). The setup process is pretty much the same.

1. Get the dataset. Follow the steps in `data/README.md`. 

2. Install cuda 10.0 if it's not available already. 

3. Install anaconda if it's not available already, and create a new environment. You need to install a few things, namely, pytorch 1.2, torchvision (*from the layers branch, which has ROI pooling*), and allennlp.

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name r2c python=3.6
source activate r2c

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg

conda install pytorch=1.2.0 -c pytorch
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm


# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

4. Now to set up the environment, run `source activate r2c && export PYTHONPATH={path_to_this_repo}`.

# Train/Evaluate models
Please refer to `models/README.md`.

### Bibtex
```
@inproceedings{LinNeurIPS2019,
  author = {J. Lin and U. Jain and A.~G. Schwing},
  title = {{TAB-VCR: Tags and Attributes based VCR Baselines}},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2019},
}
```