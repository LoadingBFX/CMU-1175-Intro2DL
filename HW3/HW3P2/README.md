# Project Name

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Optimal Methods](#optimal-methods)
4. [Configuration File Overview](#configuration-file-overview)
5. [How to Run](#how-to-run)
6. [Training Logs](#training-logs)
7. [Additional Information](#additional-information)

---

### 1. Introduction
cd /mnt/c/Users/Fanxing/CMU/1175/HW3/HW3P2
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

