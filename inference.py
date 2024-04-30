import torch
import torch.nn as nn
import argparse
import numpy as np
from lstm import SimpleLSTM
import os

def load_features(feature_path, target_length=324, feature_dim=40):
    with open(feature_path, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])

    if matrix.shape[0] > target_length:
        matrix = matrix[:target_length, :]
    elif matrix.shape[0] < target_length:
        padding = np.zeros((target_length - matrix.shape[0], feature_dim))
        matrix = np.vstack((matrix, padding))

    return matrix.T  # (40, 324)

def inference(model, device, feature_path, output_file):
    model.eval()
    features = load_features(feature_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(features)
        outputs = torch.sigmoid(outputs)  # Assuming binary classification

    # Prepare to save results with additional information
    results = outputs.cpu().numpy()
    header_info = f'Evaluation of file: {os.path.basename(feature_path)}\n0 : fake, 1 : real\n'
    np.savetxt(output_file, results, fmt='%f', header=header_info, comments='')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(feat_dim=40, time_dim=324, mid_dim=30, out_dim=1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    print(f"Model loaded from {args.model}")
    print(f"Starting inference on {args.in_file}")
    print(f"Results will be saved to {args.out}")

    inference(model, device, args.in_file, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference with an LSTM model.")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--in', type=str, dest='in_file', required=True)
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()
    main(args)
