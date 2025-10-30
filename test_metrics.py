#!/usr/bin/env python3
"""
Unit test to debug the metrics calculation in log_regression.py
"""
import torch
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

def test_metrics_calculation():
    """Test the metrics calculation logic"""
    
    # Simulate what we expect from the data loader
    print("Testing metrics calculation...")
    
    # Simulate predictions and targets
    all_preds = []
    all_probas = []
    all_targets = []
    
    # Simulate batch processing - ALL batches should have same size
    for i in range(3):
        # Simulate batch output
        batch_size = 2
        preds = torch.randint(0, 2, (batch_size,))  # Binary predictions
        probas = torch.rand(batch_size, 2)  # Two class probabilities
        probas = torch.softmax(probas, dim=1)
        
        # Try different target formats
        if i == 0:
            # Tensor targets (most common case)
            targets = torch.randint(0, 2, (batch_size,))
        elif i == 1:
            # List with multiple elements
            targets = [0, 1]  # List
        else:
            # Single element list - THIS IS THE PROBLEMATIC CASE
            targets = [1]  # Single element list
        
        all_preds.append(preds.cpu())
        all_probas.append(probas.cpu())
        
        # Handle both tensor and list targets using the SAME logic as the code
        if targets is not None:
            if isinstance(targets, torch.Tensor):
                # Ensure targets shape matches preds (handle scalar tensors)
                if targets.dim() == 0:
                    # Scalar tensor - expand to match batch
                    all_targets.append(targets.unsqueeze(0).expand_as(preds).cpu())
                else:
                    all_targets.append(targets.cpu())
            elif isinstance(targets, (list, tuple)):
                # Convert to tensor with proper shape
                targets_tensor = torch.tensor(targets)
                # Check if it's a single element that needs to be expanded
                if targets_tensor.numel() == 1:
                    # Single element - expand to match batch size
                    targets_tensor = targets_tensor.expand_as(preds)
                all_targets.append(targets_tensor.cpu())
            else:
                # Scalar value - expand to match batch
                scalar_tensor = torch.tensor(targets)
                if scalar_tensor.numel() == 1:
                    all_targets.append(scalar_tensor.expand_as(preds).cpu())
                else:
                    all_targets.append(scalar_tensor.cpu())
    
    print(f"all_preds: {[p.shape for p in all_preds]}")
    print(f"all_targets: {[t.shape for t in all_targets]}")
    print(f"all_probas: {[p.shape for p in all_probas]}")
    
    # Concatenate
    all_preds_cat = torch.cat(all_preds)
    all_targets_cat = torch.cat(all_targets)
    all_probas_cat = torch.cat(all_probas)
    
    print(f"\nConcatenated shapes:")
    print(f"all_preds_cat: {all_preds_cat.shape}")
    print(f"all_targets_cat: {all_targets_cat.shape}")
    print(f"all_probas_cat: {all_probas_cat.shape}")
    
    # Convert to numpy
    preds_np = all_preds_cat.numpy()
    targets_np = all_targets_cat.numpy()
    probas_np = all_probas_cat.numpy()[:, 1]  # Probabilities for positive class
    
    print(f"\nNumPy arrays:")
    print(f"preds_np: {preds_np}")
    print(f"targets_np: {targets_np}")
    print(f"probas_np: {probas_np}")
    
    # Calculate metrics
    try:
        balanced_acc = balanced_accuracy_score(targets_np, preds_np)
        mcc = matthews_corrcoef(targets_np, preds_np)
        auc_roc = roc_auc_score(targets_np, probas_np)
        f1 = f1_score(targets_np, preds_np)
        precision = precision_score(targets_np, preds_np)
        recall = recall_score(targets_np, preds_np)
        
        print(f"\nMetrics calculated successfully:")
        print(f"balanced_accuracy: {balanced_acc}")
        print(f"mcc: {mcc}")
        print(f"auc_roc: {auc_roc}")
        print(f"f1: {f1}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        
        return True
    except Exception as e:
        print(f"\nError calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_linear_metrics_calculation():
    """Test the metrics calculation logic for linear.py"""
    print("\n" + "="*50)
    print("Testing metrics calculation for linear.py")
    print("="*50)
    
    # Simulate accumulated_results format
    accumulated_results = {
        "classifier_1": {
            "preds": torch.rand(6, 2),
            "target": torch.randint(0, 2, (6,))
        }
    }
    
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score
    )
    
    try:
        for classifier_string, classifier_results in accumulated_results.items():
            if "preds" in classifier_results and "target" in classifier_results:
                preds_tensor = classifier_results["preds"]
                targets_tensor = classifier_results["target"]
                
                # Convert to numpy
                preds = preds_tensor.cpu().numpy()
                targets = targets_tensor.cpu().numpy()
                
                print(f"\nShapes: preds={preds.shape}, targets={targets.shape}")
                
                # Ensure shapes are compatible
                if len(preds.shape) == 2 and preds.shape[1] == 2:
                    # Convert probabilities to binary predictions
                    preds_binary = preds.argmax(axis=1)
                    probas_pos = preds[:, 1]  # Probabilities for positive class
                else:
                    # Already binary predictions
                    preds_binary = preds
                    probas_pos = preds.astype(float)
                
                # Ensure targets match preds length
                if targets.ndim > 1:
                    targets = targets.flatten()
                if preds_binary.ndim > 1:
                    preds_binary = preds_binary.flatten()
                
                # Truncate to same length if needed
                min_len = min(len(targets), len(preds_binary))
                if min_len > 0:
                    targets = targets[:min_len]
                    preds_binary = preds_binary[:min_len]
                    probas_pos = probas_pos[:min_len]
                    
                    # Calculate binary metrics
                    balanced_acc = balanced_accuracy_score(targets, preds_binary)
                    mcc = matthews_corrcoef(targets, preds_binary)
                    auc_roc = roc_auc_score(targets, probas_pos)
                    f1 = f1_score(targets, preds_binary)
                    precision = precision_score(targets, preds_binary)
                    recall = recall_score(targets, preds_binary)
                    
                    print(f"\nMetrics calculated successfully:")
                    print(f"balanced_accuracy: {balanced_acc}")
                    print(f"mcc: {mcc}")
                    print(f"auc_roc: {auc_roc}")
                    print(f"f1: {f1}")
                    print(f"precision: {precision}")
                    print(f"recall: {recall}")
                    
                    return True
        
        return False
    except Exception as e:
        print(f"\nError calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_metrics_calculation()
    success2 = test_linear_metrics_calculation()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        exit(1)

