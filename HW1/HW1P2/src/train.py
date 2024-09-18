import torch
from numpy import dtype
from torch import autocast
from tqdm.auto import tqdm
import torchaudio.transforms as T
from torch import GradScaler

time_mask = T.TimeMasking(time_mask_param=10)
freq_mask = T.FrequencyMasking(freq_mask_param=3)

def augment_data(frames):
    frames = time_mask(frames)
    frames = freq_mask(frames)
    return frames


def train(model, dataloader, optimizer, criterion, device='cpu'):
    scaler = GradScaler()
    model.train()
    tloss, tacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, (frames, phonemes) in enumerate(dataloader):
        # Apply augmentation
        # frames = augment_data(frames)

        ### Initialize Gradients
        optimizer.zero_grad()

        ### Move Data to Device (Ideally GPU)
        frames      = frames.to(device)
        phonemes    = phonemes.to(device)
        # Ensure target is of type torch.long
        phonemes = phonemes.long()

        # ### Forward Propagation
        # logits = model(frames)
        #
        # ### Loss Calculation
        # loss = criterion(logits, phonemes)

        # ### Backward Propagation
        # loss.backward()

        ### Gradient Descent
        # optimizer.step()

        # mixed precision
        with autocast(device_type ='cuda', dtype=torch.float16):
            ### Forward Propagation
            logits  = model(frames)

            ### Loss Calculation
            loss    = criterion(logits, phonemes)

        # # mixed precision
        scaler.scale(loss).backward()

        # mixed precision
        scaler.step(optimizer)
        scaler.update()

        tloss   += loss.item()
        tacc    += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc



