"""
Evaluate trained model on test set.
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    data_dir = Path('data/processed')
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"Test shape: {X_test.shape}")
    
    # Load model
    print("\nLoading trained model...")
    model_path = 'models/crnn_best.h5'
    model = keras.models.load_model(model_path)
    
    # Predict
    print("\nMaking predictions...")
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=7)
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Label names
    label_names = [
        'Normal',
        'Asthma',
        'COPD',
        'Pneumonia',
        'Bronchitis',
        'Tuberculosis',
        'Long-COVID'
    ]
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, label_names, save_path=results_dir / 'confusion_matrix.png')
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test_cat, y_pred_proba, average='macro', multi_class='ovr')
        print(f"\nROC-AUC Score (macro): {roc_auc:.4f}")
    except Exception as e:
        print(f"\nCould not calculate ROC-AUC: {e}")
    
    # Per-class accuracy
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)
    
    for i, label in enumerate(label_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"{label:15s}: {acc:.4f} ({mask.sum()} samples)")
    
    # Overall accuracy
    overall_acc = (y_pred == y_test).sum() / len(y_test)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
