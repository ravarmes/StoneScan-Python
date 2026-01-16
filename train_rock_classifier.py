# -*- coding: utf-8 -*-
"""
Rock Image Classification Training Script
==========================================

Scientific training pipeline for ornamental rock classification using:
- Transfer Learning with ResNet18
- Stratified K-Fold Cross-Validation
- Extensive Data Augmentation
- Grid Search Hyperparameter Optimization

Author: Generated for scientific publication
Date: 2026-01-15
"""

import os
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for training parameters."""
    
    # Dataset paths
    DATASET_PATH = Path(__file__).parent / "dataset"
    RESULTS_PATH = Path(__file__).parent / "results"
    
    # Class mapping (folder_name -> display_name)
    CLASS_MAPPING = {
        'g-03': 'Granito Branco Itaúnas',
        'm-02': 'Mármore Matarazzo',
        'q-01': 'Quartzito Perla',
        'q-02': 'Quartzito Wakanda',
        'q-03': 'Quartzito Verde Gaya'
    }
    
    # Folders to exclude from dataset
    EXCLUDE_FOLDERS = ['backup_originals']
    
    # Image settings
    IMAGE_SIZE = 224
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Default training parameters
    DEFAULT_EPOCHS = 25
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_N_FOLDS = 5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 7
    
    # Grid search parameters
    GRID_SEARCH_PARAMS = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [8, 16, 32],
        'optimizer': ['Adam', 'SGD', 'AdamW']
    }
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


# ============================================================================
# DATASET CLASS
# ============================================================================

class RockDataset(Dataset):
    """Custom Dataset for rock images."""
    
    def __init__(
        self,
        root_dir: Path,
        class_mapping: Dict[str, str],
        transform: Optional[transforms.Compose] = None,
        exclude_folders: List[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class folders
            class_mapping: Dict mapping folder names to class names
            transform: Torchvision transforms to apply
            exclude_folders: Folder names to exclude (e.g., backup folders)
        """
        self.root_dir = Path(root_dir)
        self.class_mapping = class_mapping
        self.transform = transform
        self.exclude_folders = exclude_folders or []
        
        # Create class to index mapping
        self.classes = list(class_mapping.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths with their labels."""
        samples = []
        
        for class_folder in self.classes:
            class_path = self.root_dir / class_folder
            if not class_path.exists():
                print(f"Warning: Class folder {class_folder} not found")
                continue
            
            class_idx = self.class_to_idx[class_folder]
            
            for item in class_path.iterdir():
                # Skip excluded folders
                if item.is_dir():
                    if any(excl in item.name for excl in self.exclude_folders):
                        continue
                    continue
                
                # Check if it's a valid image file
                if item.suffix.lower() in Config.IMAGE_EXTENSIONS:
                    samples.append((item, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self) -> np.ndarray:
        """Return all labels for stratification."""
        return np.array([label for _, label in self.samples])
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return the distribution of samples per class."""
        labels = self.get_labels()
        distribution = {}
        for class_name, class_idx in self.class_to_idx.items():
            distribution[class_name] = int(np.sum(labels == class_idx))
        return distribution


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with extensive data augmentation.
    
    These transforms help prevent overfitting by artificially
    expanding the training dataset with variations.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Only applies necessary preprocessing for inference.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class RockClassifier(nn.Module):
    """
    Rock classification model using transfer learning with ResNet18.
    
    Architecture:
    - ResNet18 backbone (pretrained on ImageNet, frozen)
    - Custom classification head with dropout regularization
    """
    
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super(RockClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_score: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = val_score
            self.best_model_state = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def get_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset handling.
    
    Uses inverse frequency weighting.
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(weights)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float
) -> optim.Optimizer:
    """Get optimizer by name."""
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model.
    
    Returns:
        Tuple of (loss, accuracy, predictions, true_labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    class_weights: torch.Tensor,
    learning_rate: float,
    optimizer_name: str,
    epochs: int,
    device: torch.device,
    patience: int = 7
) -> Dict:
    """
    Train a single fold.
    
    Returns:
        Dictionary with training history and best metrics
    """
    # Initialize model
    model = RockClassifier(num_classes=num_classes).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = get_optimizer(model, optimizer_name, learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_preds = None
    best_labels = None
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = val_preds
            best_labels = val_labels
        
        # Early stopping check
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_preds': best_preds,
        'best_labels': best_labels,
        'model_state': model.state_dict(),
        'epochs_trained': len(history['train_loss'])
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(
    dataset: RockDataset,
    n_folds: int,
    learning_rate: float,
    batch_size: int,
    optimizer_name: str,
    epochs: int,
    device: torch.device,
    patience: int = 7,
    verbose: bool = True
) -> Dict:
    """
    Run stratified K-Fold cross-validation.
    
    Returns:
        Dictionary with aggregated results from all folds
    """
    labels = dataset.get_labels()
    num_classes = len(dataset.classes)
    class_weights = get_class_weights(labels, num_classes)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    
    fold_results = []
    all_val_preds = []
    all_val_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{n_folds}")
            print(f"{'='*50}")
        
        # Create data subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Apply different transforms
        # Create separate datasets with appropriate transforms
        train_dataset_fold = RockDataset(
            dataset.root_dir,
            dataset.class_mapping,
            transform=get_train_transforms(),
            exclude_folders=dataset.exclude_folders
        )
        val_dataset_fold = RockDataset(
            dataset.root_dir,
            dataset.class_mapping,
            transform=get_val_transforms(),
            exclude_folders=dataset.exclude_folders
        )
        
        train_subset = Subset(train_dataset_fold, train_idx)
        val_subset = Subset(val_dataset_fold, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Train fold
        result = train_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            class_weights=class_weights,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            epochs=epochs,
            device=device,
            patience=patience
        )
        
        fold_results.append(result)
        all_val_preds.extend(result['best_preds'])
        all_val_labels.extend(result['best_labels'])
        
        if verbose:
            print(f"Fold {fold + 1} - Best Val Accuracy: {result['best_val_acc']:.4f}")
            print(f"Epochs trained: {result['epochs_trained']}")
    
    # Aggregate results
    val_accuracies = [r['best_val_acc'] for r in fold_results]
    
    return {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(val_accuracies),
        'std_accuracy': np.std(val_accuracies),
        'all_preds': np.array(all_val_preds),
        'all_labels': np.array(all_val_labels)
    }


# ============================================================================
# GRID SEARCH
# ============================================================================

def grid_search(
    dataset: RockDataset,
    param_grid: Dict,
    n_folds: int,
    epochs: int,
    device: torch.device,
    patience: int = 7
) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform grid search over hyperparameters.
    
    Returns:
        Tuple of (best_params, results_dataframe)
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nGrid Search: {len(combinations)} combinations to evaluate")
    print(f"Total training runs: {len(combinations) * n_folds}")
    print("-" * 60)
    
    results = []
    best_accuracy = 0
    best_params = None
    best_cv_result = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        cv_result = run_cross_validation(
            dataset=dataset,
            n_folds=n_folds,
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            optimizer_name=params['optimizer'],
            epochs=epochs,
            device=device,
            patience=patience,
            verbose=False
        )
        
        mean_acc = cv_result['mean_accuracy']
        std_acc = cv_result['std_accuracy']
        
        print(f"Result: {mean_acc:.4f} ± {std_acc:.4f}")
        
        results.append({
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'optimizer': params['optimizer'],
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        })
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_params = params
            best_cv_result = cv_result
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_accuracy', ascending=False)
    
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best Parameters: {best_params}")
    print(f"Best Mean Accuracy: {best_accuracy:.4f}")
    
    return best_params, results_df, best_cv_result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_learning_curves(
    fold_results: List[Dict],
    save_path: Path
):
    """Plot learning curves from all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss curves
    for i, result in enumerate(fold_results):
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], '--', alpha=0.5, label=f'Fold {i+1} Train')
        axes[0].plot(epochs, history['val_loss'], '-', alpha=0.7, label=f'Fold {i+1} Val')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy curves
    for i, result in enumerate(fold_results):
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], '--', alpha=0.5, label=f'Fold {i+1} Train')
        axes[1].plot(epochs, history['val_acc'], '-', alpha=0.7, label=f'Fold {i+1} Val')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to: {save_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: Dict[str, str],
    idx_to_class: Dict[int, str],
    save_path: Path
):
    """Generate and save detailed classification report."""
    # Get class names in order
    target_names = [class_mapping[idx_to_class[i]] for i in range(len(class_mapping))]
    
    # Generate report
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    
    # Calculate additional metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create full report
    full_report = f"""
================================================================================
CLASSIFICATION REPORT - ROCK IMAGE CLASSIFICATION
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL METRICS:
----------------
Accuracy:          {accuracy:.4f}
Weighted Precision:{precision:.4f}
Weighted Recall:   {recall:.4f}
Weighted F1-Score: {f1:.4f}
Cohen's Kappa:     {kappa:.4f}

DETAILED CLASSIFICATION REPORT:
-------------------------------
{report}

================================================================================
LATEX TABLE FORMAT (for article):
================================================================================
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Classe}} & \\textbf{{Precisão}} & \\textbf{{Revocação}} & \\textbf{{F1-Score}} & \\textbf{{Suporte}} \\\\
\\hline
"""
    
    # Add per-class metrics to LaTeX table
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    for i, name in enumerate(target_names):
        full_report += f"{name} & {precision_per_class[i]:.4f} & {recall_per_class[i]:.4f} & {f1_per_class[i]:.4f} & {support_per_class[i]} \\\\ \n\\hline\n"
    
    full_report += f"""\\textbf{{Média Ponderada}} & {precision:.4f} & {recall:.4f} & {f1:.4f} & {sum(support_per_class)} \\\\
\\hline
\\end{{tabular}}
\\caption{{Métricas de classificação por classe de rocha.}}
\\label{{tab:classification_metrics}}
\\end{{table}}
================================================================================
"""
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"Classification report saved to: {save_path}")
    return full_report


def save_training_summary(
    best_params: Dict,
    cv_result: Dict,
    grid_results_df: pd.DataFrame,
    class_distribution: Dict,
    save_dir: Path
):
    """Save comprehensive training summary."""
    # Save grid search results
    grid_results_df.to_csv(save_dir / 'grid_search_results.csv', index=False)
    
    # Create summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_hyperparameters': best_params,
        'cross_validation_results': {
            'mean_accuracy': float(cv_result['mean_accuracy']),
            'std_accuracy': float(cv_result['std_accuracy']),
            'fold_accuracies': [float(r['best_val_acc']) for r in cv_result['fold_results']]
        },
        'dataset_info': {
            'total_samples': sum(class_distribution.values()),
            'class_distribution': class_distribution
        },
        'training_config': {
            'model': 'ResNet18 (pretrained)',
            'image_size': Config.IMAGE_SIZE,
            'n_folds': len(cv_result['fold_results']),
            'early_stopping_patience': Config.EARLY_STOPPING_PATIENCE
        }
    }
    
    with open(save_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Training summary saved to: {save_dir / 'training_summary.json'}")


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

def train_final_model(
    dataset: RockDataset,
    best_params: Dict,
    epochs: int,
    device: torch.device,
    save_path: Path
) -> nn.Module:
    """
    Train final model on entire dataset with best hyperparameters.
    """
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 60)
    
    # Create dataset with training transforms
    full_dataset = RockDataset(
        dataset.root_dir,
        dataset.class_mapping,
        transform=get_train_transforms(),
        exclude_folders=dataset.exclude_folders
    )
    
    # Create data loader
    train_loader = DataLoader(
        full_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    num_classes = len(dataset.classes)
    model = RockClassifier(num_classes=num_classes).to(device)
    
    # Class weights
    labels = dataset.get_labels()
    class_weights = get_class_weights(labels, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, best_params['optimizer'], best_params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels_batch in pbar:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
            
            pbar.set_postfix({
                'loss': f"{running_loss/total:.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
    
    print(f"\nFinal model saved to: {save_path}")
    return model


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Rock Image Classification Training Script'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(Config.DATASET_PATH),
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.DEFAULT_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=Config.DEFAULT_N_FOLDS,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--no-grid-search',
        action='store_true',
        help='Skip grid search and use default parameters'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Quick test run with minimal parameters'
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    print("=" * 60)
    print("ROCK IMAGE CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Dataset path: {args.data_dir}")
    
    # Create results directory
    results_dir = Path(args.data_dir) / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Initialize dataset (no transforms for scanning)
    dataset = RockDataset(
        root_dir=Path(args.data_dir),
        class_mapping=Config.CLASS_MAPPING,
        transform=None,
        exclude_folders=Config.EXCLUDE_FOLDERS
    )
    
    # Print dataset info
    print(f"\nDataset Summary:")
    print("-" * 40)
    class_dist = dataset.get_class_distribution()
    for folder, name in Config.CLASS_MAPPING.items():
        count = class_dist.get(folder, 0)
        print(f"  {name}: {count} images")
    print(f"  Total: {len(dataset)} images")
    
    # Adjust parameters for dry run
    if args.dry_run:
        print("\n*** DRY RUN MODE ***")
        args.epochs = 2
        args.folds = 2
        Config.GRID_SEARCH_PARAMS = {
            'learning_rate': [0.001],
            'batch_size': [16],
            'optimizer': ['Adam']
        }
    
    # Grid Search or use defaults
    if args.no_grid_search:
        best_params = {
            'learning_rate': Config.DEFAULT_LEARNING_RATE,
            'batch_size': Config.DEFAULT_BATCH_SIZE,
            'optimizer': 'Adam'
        }
        print(f"\nUsing default parameters: {best_params}")
        
        # Run cross-validation with default params
        cv_result = run_cross_validation(
            dataset=dataset,
            n_folds=args.folds,
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            optimizer_name=best_params['optimizer'],
            epochs=args.epochs,
            device=Config.DEVICE,
            patience=Config.EARLY_STOPPING_PATIENCE
        )
        grid_results_df = pd.DataFrame([{
            **best_params,
            'mean_accuracy': cv_result['mean_accuracy'],
            'std_accuracy': cv_result['std_accuracy']
        }])
    else:
        # Perform grid search
        best_params, grid_results_df, cv_result = grid_search(
            dataset=dataset,
            param_grid=Config.GRID_SEARCH_PARAMS,
            n_folds=args.folds,
            epochs=args.epochs,
            device=Config.DEVICE,
            patience=Config.EARLY_STOPPING_PATIENCE
        )
    
    # Print cross-validation results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {cv_result['mean_accuracy']:.4f} ± {cv_result['std_accuracy']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        cv_result['all_labels'],
        cv_result['all_preds'],
        [Config.CLASS_MAPPING[dataset.idx_to_class[i]] for i in range(len(Config.CLASS_MAPPING))],
        results_dir / 'confusion_matrix.png'
    )
    
    # Learning curves
    plot_learning_curves(
        cv_result['fold_results'],
        results_dir / 'learning_curves.png'
    )
    
    # Classification report
    generate_classification_report(
        cv_result['all_labels'],
        cv_result['all_preds'],
        Config.CLASS_MAPPING,
        dataset.idx_to_class,
        results_dir / 'classification_report.txt'
    )
    
    # Save training summary
    save_training_summary(
        best_params,
        cv_result,
        grid_results_df,
        class_dist,
        results_dir
    )
    
    # Train final model on full dataset
    final_model = train_final_model(
        dataset=dataset,
        best_params=best_params,
        epochs=args.epochs,
        device=Config.DEVICE,
        save_path=results_dir / 'best_model.pth'
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    for f in results_dir.iterdir():
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
