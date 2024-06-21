#!python

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from unet import UNet
from dataset import LineDataset
from sklearn.metrics import average_precision_score, recall_score
from augment import LineAugment
from utils import select_device
import argparse

torch.backends.mps.enabled = True

def train(model, train_loader, val_loader, save_path, pretrained_path, num_epochs, lr):
    device = select_device()
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if pretrained_path is not None:
        # Load pretrained model
        model.load_state_dict(torch.load(pretrained_path))
        print("Loaded pretrained model from:", pretrained_path)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        
        # Training loop
        model.train()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Valid / 10 rounds
        if (epoch + 1) % 10 == 0:
            val_loss = 0.0
            val_ap = 0.0
            val_ar = 0.0
            
            model.eval()
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    masks = masks.unsqueeze(1)
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
                    
                    # AP and AR
                    outputs = outputs.detach().cpu().numpy()
                    masks = masks.squeeze(1).detach().cpu().numpy() # remove the channel dim
                    for i in range(len(outputs)):
                        ap = average_precision_score(masks[i].flatten(), outputs[i].flatten())
                        ar = recall_score(masks[i].flatten(), outputs[i].flatten() > 0.6)
                        val_ap += ap
                        val_ar += ar
            
            val_loss /= len(val_loader.dataset)
            val_ap /= len(val_loader.dataset)
            val_ar /= len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val AP: {val_ap:.4f}, Val AR: {val_ar:.4f}")
            
            # Save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_path, f"best_AP{val_ap:.2f}_AR{val_ar:.2f}.pth"))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Training a model')
    parser.add_argument('--train_data', default='./datasets/train', help='train datasets path')
    parser.add_argument('--val_data', default='./datasets/val', help='valid datasets path')
    parser.add_argument('--save_path', default='./saved_model/', help='output model save path')
    parser.add_argument('--pretrained_path', default=None, help='pretrained model path')
    parser.add_argument('--num_epochs', default=100, help='number of epochs', type=int)
    parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--batch_size', default=4, help='batch size', type=int)
    parser.add_argument('--imgsz', default=512, help='image size', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    imgsz = args.imgsz
    train_data_dir = args.train_data
    val_data_dir = args.val_data
    save_path = args.save_path
    pretrained_path = args.pretrained_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr

    augment = LineAugment(p=0.5)
    # augment = None

    # Prepare datasets
    train_dataset = LineDataset(train_data_dir, (imgsz, imgsz), augment=augment)
    val_dataset = LineDataset(val_data_dir, (imgsz, imgsz), augment=augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = UNet(n_channels=3, n_classes=1)

    # Training
    train(model, train_loader, val_loader, save_path, pretrained_path, num_epochs=num_epochs, lr=lr)

if __name__ == '__main__':
    main()