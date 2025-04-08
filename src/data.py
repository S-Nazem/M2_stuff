import re
import numpy as np
import torch
import pandas as pd

def extract_numerical_values(decoded_string):
    try:
        parsed_array = np.array([
            list(map(float, item.split(",")))
            for item in decoded_string.split(";") if "," in item
        ])
        if parsed_array.shape[0] >= 40:
            return parsed_array[:40] * 10
        else:
            return np.zeros((40, 2))
    except ValueError:
        return np.zeros((40, 2))


def run_forecast(model, tokenizer, input_ids, iterations):
    model.eval()
    all_outputs = []
    mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        for _ in range(iterations):
            generated = model.generate(
                input_ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=1100,
                min_length=1000
            )
            new_tokens = generated[0].tolist()[len(input_ids[0]):]
            decoded_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            numeric_values = extract_numerical_values(decoded_text)
            all_outputs.append(numeric_values)

    all_outputs = np.array(all_outputs)
    return np.mean(all_outputs, axis=0) if iterations > 1 else all_outputs[0]



# Function to compute convergence speed
def compute_convergence_speed(train_loss_csv):
    df = pd.read_csv(train_loss_csv)
    losses = df["train_loss"].values
    window = 50  # Moving window to smooth noise
    
    # Compute moving average to reduce variance
    smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')

    # Find first step where loss stops decreasing significantly
    for i in range(1, len(smoothed_loss)):
        if (smoothed_loss[i-1] - smoothed_loss[i]) / smoothed_loss[i-1] < 0.0001:  # <5% loss reduction
            return i  # Return convergence step
    
    return len(losses)  # If no clear convergence, return last step




# Tokenize dataset with sliding windows
def process_sequences(texts, tokenizer, max_length=512, stride=256):
    all_input_ids = []
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]

        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat([chunk, torch.full((max_length - len(chunk),), tokenizer.pad_token_id)])
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)