import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm


def generate_submission(checkpoint_path, test_loader, model, device, phonemes, submission_file="submission.csv"):
    """
    Load a model from a checkpoint and generate a submission CSV file with predictions.

    Parameters:
        checkpoint_path (str): Path to the model checkpoint file.
        test_loader (DataLoader): DataLoader for test data.
        model_class (torch.nn.Module): The model class used for instantiation.
        device (torch.device): Device to run the model on (CPU or GPU).
        phonemes (list): List of phonemes for prediction mapping.
        submission_file (str): The name of the output CSV file.
    """
    # Check if the checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}.")
        return

    # Load the model state
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    test_predictions = []

    # Make predictions on the test set
    ## Which mode do you need to avoid gradients?
    with torch.no_grad():

        for i, mfccs in enumerate(tqdm(test_loader)):
            mfccs = mfccs.to(device)

            logits = model(mfccs)

            ### Get most likely predicted phoneme with argmax
            predicted_phonemes = torch.argmax(logits, dim=1)

            test_predictions.extend(predicted_phonemes.cpu().numpy())

    # Write predictions to a CSV file
    with open(submission_file, "w") as f:
        f.write("id,label\n")
        for i, pred in enumerate(test_predictions):
            predicted_phoneme = phonemes[pred]  # Map index to phoneme
            f.write(f"{i},{predicted_phoneme}\n")

    print(f"Submission file '{submission_file}' generated successfully.")

