import torch
import wandb

from HW1.HW1P2.models.network import Network
from HW1.HW1P2.src.train import train, test, eval
from data.audiodataset import AudioDataset
from data.audiotestdataset import  AudioTestDataset
from torchsummaryX import summary
import gc

if __name__ == '__main__':

    DATASET_PATH = "C:\\Users\\Fanxing\\CMU\\CMU-1175-Intro2DL\\HW1\\HW1P2\\11785-f24-hw1p2"
    print(DATASET_PATH)

    ### PHONEME LIST
    PHONEMES = [
        '[SIL]', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
        'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
        'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
        'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
        'V', 'W', 'Y', 'Z', 'ZH', '[SOS]', '[EOS]']

    config = {
        'epochs'        : 5,
        'batch_size'    : 2048,
        'context'       : 20,
        'init_lr'       : 1e-3,
    }
    print(config)

    """# Create Datasets"""
    print("Loading data...")
    TRAIN_DATASIZE = -1
    VAL_DATASIZE = -1
    train_data = AudioDataset(DATASET_PATH, TRAIN_DATASIZE, PHONEMES, config['context'], 'train-clean-100')
    val_data = AudioDataset(DATASET_PATH, VAL_DATASIZE, PHONEMES, config['context'], 'dev-clean')
    test_data = AudioTestDataset(DATASET_PATH, config['context'], 'test-clean')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=4,
        batch_size=config['batch_size'],
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=2,
        batch_size=config['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=config['batch_size'],
        pin_memory=True,
        shuffle=False
    )

    print("Batch size     : ", config['batch_size'])
    print("Context        : ", config['context'])
    print("Input size     : ", (2 * config['context'] + 1) * 28)
    print("Output symbols : ", len(PHONEMES))

    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    for i, data in enumerate(train_loader):
        frames, phoneme = data
        print(frames.shape, phoneme.shape)
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    INPUT_SIZE = (2 * config['context'] + 1) * 28  # Why is this the case?
    model = Network(INPUT_SIZE, len(train_data.phonemes)).to(device)
    summary(model, frames.to(device))

    criterion = torch.nn.CrossEntropyLoss()  # Defining Loss function.

    optimizer = torch.optim.Adam(model.parameters(), lr=config['init_lr'])  # Defining Optimizer

    torch.cuda.empty_cache()
    gc.collect()

    wandb.login(
        key="46b9373c96fe8f8327255e7da8a4046da7ffeef6")  # API Key is in your wandb account, under settings (wandb.ai/settings)

    # Create your wandb run
    run = wandb.init(
        name="first-run",  ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit=True,  ### Allows reinitalizing runs when you re-run this cell
        # id     = "y28t31uz", ### Insert specific run id here if you want to resume a previous run
        # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw1p2",  ### Project should be created in your wandb account
        config=config  ### Wandb Config for your run
    )

    ### Save your model architecture as a string with str(model)
    model_arch = str(model)

    ### Save it in a txt file
    arch_file = open("model_arch.txt", "w")
    file_write = arch_file.write(model_arch)
    arch_file.close()

    ### log it in your wandb run with wandb.save()
    wandb.save('model_arch.txt')

    """# Experiment

    Now, it is time to finally run your ablations! Have fun!
    """

    # Iterate over number of epochs to train and evaluate your model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.watch(model, log="all")

    for epoch in range(config['epochs']):
        print("\nEpoch {}/{}".format(epoch + 1, config['epochs']))

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval(model, val_loader, criterion, device)

        print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc * 100, train_loss,
                                                                                        curr_lr))
        print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc * 100, val_loss))

        ### Log metrics at each epoch in your run
        # Optionally, you can log at each batch inside train/eval functions
        # (explore wandb documentation/wandb recitation)
        wandb.log({'train_acc': train_acc * 100, 'train_loss': train_loss,
                   'val_acc': val_acc * 100, 'valid_loss': val_loss, 'lr': curr_lr})
        predictions = test(model, test_loader, device)

    ### Create CSV file with predictions
    with open("./submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(predictions)):
            predicted_phoneme = PHONEMES[predictions[i]]
            f.write("{},{}\n".format(i, predicted_phoneme))

    ### Finish your wandb run
    run.finish()

### Submit to kaggle competition using kaggle API (Uncomment below to use)
# !kaggle competitions submit -c 11785-hw1p2-f24 -f ./submission.csv -m "Test Submission"