import torch
from tqdm.auto import tqdm


def test(model, test_loader, device='cpu'):
    ### What you call for model to perform inference?
    model.eval()

    ### List to store predicted phonemes of test datasets
    test_predictions = []

    ### Which mode do you need to avoid gradients?
    with torch.no_grad():

        for i, mfccs in enumerate(tqdm(test_loader)):

            mfccs   = mfccs.to(device)

            logits  = model(mfccs)

            ### Get most likely predicted phoneme with argmax
            predicted_phonemes = torch.argmax(logits, dim=1)

            test_predictions.extend(predicted_phonemes.cpu().numpy())

    return test_predictions