# HW1P2 - Phoneme Recognition

This repository contains the solution for HW1P2, a phoneme recognition task using deep learning. The project is implemented in PyTorch and includes a complete pipeline for training, evaluation, and testing of the phoneme recognition model.

## File Structure

```plaintext
HW1P2/
│
├── checkpoints/                # Directory to save model checkpoints
├── config/                     # Directory for configuration files
├── data/                       # Directory for dataset storage
├── datasets/                   # Dataset handling scripts
│   ├── analysis.py             # Data analysis script
│   ├── data_loader.py          # Data loader for training/validation data
│   └── test_data_loader.py     # Data loader for test data
│
├── models/                     # Model-related files
│   └── model.py                # Model architecture
│
├── notebook/                   # Jupyter notebooks for experiments
│
├── src/                        # Source files for model training, evaluation, and testing
│   ├── eval.py                 # Evaluation script
│   ├── test.py                 # Test script
│   ├── train.py                # Training script
│   └── utils.py                # Utility functions
│
├── wandb/                      # Wandb experiment tracking folder
│   └── generate_submission.py  # Script to generate submissions
│
├── Hw1p2_Fall_2024_Writeup.pdf  # Project write-up
├── main.py                     # Main script to run the model
├── README.md                   # Project documentation
├── requirements.txt            # Python package dependencies
├── submission.csv              # Final submission file with predictions
├── submission_test_83.csv      # Additional test submission file
└── model_arch.txt              # Model architecture description
```

## How to Execute
Prerequisites
Install Dependencies:
Ensure you have Python installed. Install required packages by running:
```aiignore
pip install -r requirements.txt
```
Dataset:
Place the dataset in the data/ directory. The dataset directory should have the following structure:
```aiignore
data/
└── 11785-f24-hw1p2/
    ├── train-clean-100/    # Training data
    ├── dev-clean/      # Validation data
    └── test-clean/     # Test data

```

Run Training
To start training the phoneme recognition model, execute the main.py file. You can adjust the configuration parameters as needed.
```aiignore
python main.py
```

Parameters Configuration
In the main.py, key parameters are stored in a dictionary named config. You can modify the parameters before running the script.
```aiignore
config = {
    'model_name'    : time_stamp,     # Unique name for the model
    'epochs'        : 200,            # Number of training epochs
    'batch_size'    : 4096,           # Batch size
    'context'       : 30,             # Context window size
    'init_lr'       : 0.001,          # Initial learning rate
    'weight_decay'  : 0.01,           # Weight decay for optimizer
}

```
Resume Training: If you want to resume training from a checkpoint, set resume_training to True and specify the path to the checkpoint in the resume_from variable.

### Hyperparameters
You can set additional hyperparameters like the optimizer, learning rate schedule, and the loss function:

- **Optimizer**:  
   The project uses `AdamW` with a learning rate of 0.001 and weight decay of 0.01.

```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
```
Scheduler:
The learning rate scheduler is set to CosineAnnealingLR to gradually reduce the learning rate during training.
```aiignore
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
```
Loss Function:
The loss function used is CrossEntropyLoss:
```aiignore
criterion = torch.nn.CrossEntropyLoss()
```
### Model Summary 
```aiignore
 11 ----------------------------------------------------------------------------------------------------
 12 Layer                   Kernel Shape         Output Shape         # Params (K)      # Mult-Adds (M)
 13 ====================================================================================================
 14 0_Linear                [1708, 1024]         [4096, 1024]             1,750.02                 1.75
 15 1_BatchNorm1d                 [1024]         [4096, 1024]                 2.05                 0.00
 16 2_GELU                             -         [4096, 1024]                    -                    -
 17 3_Dropout                          -         [4096, 1024]                    -                    -
 18 4_Linear                [1024, 2048]         [4096, 2048]             2,099.20                 2.10
 19 5_BatchNorm1d                 [2048]         [4096, 2048]                 4.10                 0.00
 20 6_GELU                             -         [4096, 2048]                    -                    -
 21 7_Dropout                          -         [4096, 2048]                    -                    -
 22 8_Linear                [2048, 3000]         [4096, 3000]             6,147.00                 6.14
 23 9_BatchNorm1d                 [3000]         [4096, 3000]                 6.00                 0.00
 24 10_GELU                            -         [4096, 3000]                    -                    -
 25 11_Dropout                         -         [4096, 3000]                    -                    -
 26 12_Linear               [3000, 2048]         [4096, 2048]             6,146.05                 6.14
 27 13_BatchNorm1d                [2048]         [4096, 2048]                 4.10                 0.00
 28 14_GELU                            -         [4096, 2048]                    -                    -
 29 15_Dropout                         -         [4096, 2048]                    -                    -
 30 16_Linear               [2048, 1024]         [4096, 1024]             2,098.18                 2.10
 31 17_BatchNorm1d                [1024]         [4096, 1024]                 2.05                 0.00
 32 18_GELU                            -         [4096, 1024]                    -                    -
 33 19_Dropout                         -         [4096, 1024]                    -                    -
 34 20_Linear               [1024, 1024]         [4096, 1024]             1,049.60                 1.05
 35 21_BatchNorm1d                [1024]         [4096, 1024]                 2.05                 0.00
 36 22_GELU                            -         [4096, 1024]                    -                    -
 37 23_Dropout                         -         [4096, 1024]                    -                    -
 38 24_Linear                [1024, 512]          [4096, 512]               524.80                 0.52
 39 25_BatchNorm1d                 [512]          [4096, 512]                 1.02                 0.00
 40 26_GELU                            -          [4096, 512]                    -                    -
 41 27_Dropout                         -          [4096, 512]                    -                    -
 42 28_Linear                 [512, 256]          [4096, 256]               131.33                 0.13
 43 29_BatchNorm1d                 [256]          [4096, 256]                 0.51                 0.00
 44 30_GELU                            -          [4096, 256]                    -                    -
 45 31_Dropout                         -          [4096, 256]                    -                    -
 46 32_Linear                  [256, 42]           [4096, 42]                10.79                 0.01
 47 ====================================================================================================
 48 # Params:    19,978.83K
 49 # Mult-Adds: 19.96M
 50 ----------------------------------------------------------------------------------------------------
```