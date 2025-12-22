#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
from itertools import product
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

warnings.filterwarnings('ignore', category=UserWarning)

class Config:
    DATA_INPUT_DIR = '/kaggle/input/stonescan-rochas/dataset' 
    MODEL_SAVE_PATH = '/kaggle/working/best_model_rochas_balanced.pth'
    RANDOM_SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

class ImageListDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

def collect_dataset_items() -> Tuple[List[Tuple[str, int]], List[str]]:
    base_dir = Config.DATA_INPUT_DIR
    class_names = sorted([name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    items = []
    for cls in class_names:
        cls_dir = os.path.join(base_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    items.append((os.path.join(root, f), class_to_idx[cls]))
    return items, class_names

def split_train_val_test(items: List[Tuple[str, int]]):
    labels = [lbl for _, lbl in items]
    idx_all = np.arange(len(items))
    idx_train_val, idx_test = train_test_split(idx_all, test_size=0.15, random_state=Config.RANDOM_SEED, stratify=labels)
    labels_train_val = [labels[i] for i in idx_train_val]
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.18, random_state=Config.RANDOM_SEED, stratify=labels_train_val)
    return [items[i] for i in idx_train], [items[i] for i in idx_val], [items[i] for i in idx_test]

def create_model(num_classes: int):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    
    # Congela tudo inicialmente
    for param in model.parameters():
        param.requires_grad = False
        
    # Libera o último bloco para fine-tuning (ajuda a aprender texturas de rochas)
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(Config.DEVICE)

def train_one_run(model, train_loader, val_loader, num_epochs: int, lr: float, class_weights: torch.Tensor) -> Tuple[float, Dict]:
    # AQUI ESTÁ A CHAVE: CrossEntropy com pesos para equilibrar as classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_balanced_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validação focada em Balanced Accuracy
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x.to(Config.DEVICE))
                val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
                val_labels.extend(y.numpy())
        
        # Métrica de seleção: Balanced Accuracy é melhor para dados desbalanceados
        b_acc = balanced_accuracy_score(val_labels, val_preds)
        
        if b_acc > best_balanced_acc:
            best_balanced_acc = b_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_balanced_acc, best_state

def main():
    set_seeds()
    items, class_names = collect_dataset_items()
    train_items, val_items, test_items = split_train_val_test(items)

    # Cálculo dos Pesos das Classes (Fórmula: n_samples / (n_classes * n_samples_at_class))
    labels_train = [lbl for _, lbl in train_items]
    counts = np.bincount(labels_train)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    class_weights = torch.FloatTensor(weights).to(Config.DEVICE)
    print(f"Pesos aplicados às classes: {weights}")

    tfms = get_transforms()
    train_ds = ImageListDataset(train_items, tfms['train'])
    val_ds = ImageListDataset(val_items, tfms['eval'])
    test_ds = ImageListDataset(test_items, tfms['eval'])

    # Grid search simplificado para focar em qualidade
    grid = {
        'LR': [1e-4, 5e-5],
        'BS': [8, 16],
        'EP': [20] # Mais épocas para o fine-tuning
    }

    best_overall = {'b_acc': -1.0, 'params': None, 'state': None}

    for lr, batch_size, epochs in product(grid['LR'], grid['BS'], grid['EP']):
        print(f"\nTestando: LR={lr}, BS={batch_size}")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        model = create_model(len(class_names))
        b_acc, state = train_one_run(model, train_loader, val_loader, epochs, lr, class_weights)
        
        print(f"Balanced Acc Validação: {b_acc:.4f}")
        
        if b_acc > best_overall['b_acc']:
            best_overall.update({'b_acc': b_acc, 'params': (lr, batch_size, epochs), 'state': state})

    # Avaliação Final
    final_model = create_model(len(class_names))
    final_model.load_state_dict(best_overall['state'])
    torch.save(final_model.state_dict(), Config.MODEL_SAVE_PATH)
    
    test_loader = DataLoader(test_ds, batch_size=best_overall['params'][1])
    print('\n' + '='*30 + '\nRESULTADO FINAL NO TESTE\n' + '='*30)
    
    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            outputs = final_model(x.to(Config.DEVICE))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(y.numpy())
    
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    main()