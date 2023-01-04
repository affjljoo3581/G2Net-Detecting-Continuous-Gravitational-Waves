# G2Net Detecting Continuous Gravitational Waves

## Introduction
This repository contains the code that achieve 12th place in [G2Net Detecting Continuous Gravitational Waves](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/overview).

## Requirements
* numpy
* omegaconf
* pandas
* pyfstat
* pytorch_lightning
* scikit_learn
* timm
* torch
* tqdm
* wandb

Instead of installing the above modules independently, you can simply do at once by using:
```bash
$ pip install -f requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

This repository supports [NVIDIA Apex](https://github.com/NVIDIA/apex). It will automatically detect the apex module and if it is found then some training procedures will be replaced with the highly-optimized and fused operations in the apex module. Run the below codes in the terminal to install apex and enable performance boosting:

```bash
$ git clone https://github.com/NVIDIA/apex
$ sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
$ rm -rf apex
```

Instead, we recommend to use docker and [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) where apex, optimized cuda driver and faster pytorch kernel are installed:
```bash
$ docker run --gpus all -it nvcr.io/nvidia/pytorch:22.07-py3
```

## Getting started
First, you need pure simulated signal templates to create noise-combined images.
```bash
$ python scripts/simulate_signals.py resources/competition/timestamps.pkl
```
Note that the timestamps are extracted from test HDF5 data.

Next, generate random gaussian background noise to combine with the pure signals.
```bash
$ python scripts/synthesize_external_psds.py resources/external/train/signals
```
You can also make input data for validation.

To evaluate and predict with competition data, run the below code to convert HDF5 to our input format.
```bash
$ python extract_psds_from_hdf5.py [competition train/test hdf5 directory]
```

Now, let's train the model. It is simple!
```bash
$ python src/train.py config/convnext_small_in22ft1k.yaml
```
After training, you can find a pretrained model path like `convnext_small_in22ft1k-6f6648-last.pt`. There are two models - first one is a best-scored model and the second one is a latest one. To predict the test signals, run the below code:
```bash
$ python src/predict.py convnext_small_in22ft1k-6f6648-last.pt --use-flip-tta
```
That's all! There should be `convnext_small_in22ft1k-6f6648-last.csv` and just submit to kaggle.