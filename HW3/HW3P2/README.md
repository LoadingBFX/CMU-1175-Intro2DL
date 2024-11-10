# Project Name

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Data](#Data)
4. [Configuration File Overview](#configuration-file-overview)
5. [How to Run](#how-to-run)
6. [Result](#Result)


---

### 1. Introduction
This project is a part of the course "Deep Learning for Speech and Language" at IIIT Hyderabad. The task is to implement a speech recognition system using the CTC loss function. The model is trained on the LibriSpeech dataset and the model is evaluated on the test set provided by the course instructors. The model is trained using the CTC loss function and the model is evaluated using the CER metric. The model is trained using the Adam optimizer and the learning rate is scheduled using the ReduceLROnPlateau scheduler. The model is trained using the following hyperparameters:
- Batch Size: 32
- Learning Rate: 0.001
- Number of Epochs: 50
- Number of Layers: 5
- Number of Hidden Units: 512
- Dropout: 0.1
- Bidirectional: True
- Number of Classes: 29
- Number of Workers: 4
- Seed: 42

### 2. Environment Setup
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchtext==0.14.1 torchaudio==0.13.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu117 -q

pip install torchsummaryx==1.3.0
pip install wandb --quiet
pip install python-Levenshtein -q

git clone --recursive https://github.com/parlance/ctcdecode.git
pip install wget -q
cd ctcdecode
pip install . -q
cd ..

pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8 -q

```

### 3.Data

```bash
pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8 -q
mkdir /root/.kaggle
```
```python

with open("/root/.kaggle/kaggle.json", "w+") as f:
    f.write('{"username":"<username>","key":"<key>"}')
    # Put your kaggle username & key here
```
```bash
chmod 600 /root/.kaggle/kaggle.json

kaggle competitions download -c hw3p2-785-f24
unzip -q hw3p2-785-f24.zip
ls
```
### Configuration File Overview
The configuration file config/config.yaml contains various parameters for training, validation, and testing. Here is an example configuration:
```yaml
data_folder: './11785-f24-hw3p2/'
save_model_folder: './ckpt/'


train:
  lr: 2e-3
  epochs: 100
  warmup: 2
  batch_size: 64
  save_interval: 10
  weight_decay: 0.0001
  
scheduler:
  T_max: 100
  patience: 2

specaug:
  freq_mask_param: 10
  time_mask_param: 100

model:
  input_size: 28
  embed_size: 1024

encoder:
  expand_dims: [128, 512]
  kernel_size: 7

pBLSTMs:
  dropout_prob: 0.3

decoder:
  dropout_prob: 0.2

decode:
  beam_width: 10
```

### How to Run

1. **Set up the configuration:**  
   Edit the `config/config.yaml` file to set the appropriate parameters for training, validation, and testing.

2. **Train the model:**  
   Run the training script:  
   ```bash
   python HW3/HW3P2/train.py
    ```
   The evaluation is performed during the training process.
3. **Test the model:**  
   Run the testing script:  
   ```bash
   python HW3/HW3P2/test.py
   ```
   The model is evaluated on the test set provided by the course instructors.



### Results
Here are some visualizations of the training process:  
<img src="./log.png"></img> 
<img src="./metric.png"></img>