import re
import numpy as np
import torch
import pandas as pd

def parse_decoded_output(decoded_str, scale_factor=10):
    try:
        parsed_data = np.array([
            list(map(float, pair.split(",")))
            for pair in decoded_str.split(";") if "," in pair
        ])
        return parsed_data[:40] * scale_factor if parsed_data.shape[0] >= 40 else np.zeros((40, 2))
    except ValueError:
        return np.zeros((40, 2))
    


    
def run_prediction(model, tokenizer, input_tensor, num_runs):
    model.eval()
    predictions = []
    attention_mask = (input_tensor != tokenizer.pad_token_id).long()

    with torch.no_grad():
        for _ in range(num_runs):
            output_tokens = model.generate(input_tensor, attention_mask=attention_mask, max_new_tokens=1100, min_length=1000)
            generated_tokens = output_tokens[0].tolist()[len(input_tensor[0]):]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            prediction = parse_decoded_output(decoded_output)
            predictions.append(prediction)
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0) if num_runs > 1 else predictions[0]




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




# ✅ Tokenize dataset with sliding windows
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