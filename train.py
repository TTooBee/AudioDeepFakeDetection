import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Assuming lstm.py and Dataset.py are in the same directory or appropriately installed as modules
from lstm import SimpleLSTM
from Dataset import AudioFeaturesDataset

def train(model, device, train_loader, optimizer, criterion, epochs, save_path):
    model.train()
    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioFeaturesDataset(args.real, args.fake)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SimpleLSTM(feat_dim=40, time_dim=324, mid_dim=30, out_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    train(model, device, train_loader, optimizer, criterion, args.epochs, args.model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTM model.")
    parser.add_argument('--real', type=str, default='features_real_temp')
    parser.add_argument('--fake', type=str, default='features_fake_temp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model', type=str, default='model_weights.pt')
    
    args = parser.parse_args()
    main(args)