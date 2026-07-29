#!/usr/bin/env python3


import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.classification_evaluator import evaluate_classification_performance

def evaluate_model_classification(model, dataloader, line_type="railroads", model_name="LDTR_Model"):

    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:

            images = batch['images'].to(model.device)
            targets = batch['targets']
            
            outputs = model(images)
            
            pred_logits = outputs['pred_logits']  # Shape: [batch_size, num_queries, 4]
            
            target_node_types = targets['node_types']
            
            batch_targets = torch.zeros(pred_logits[..., 0].shape, dtype=torch.long).to(pred_logits.device)
            

            indices = model.matcher(outputs, targets)  
            
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx) > 0 and len(tgt_idx) > 0:
                    batch_targets[b, src_idx] = target_node_types[b][tgt_idx]
            
            pred_flat = pred_logits.view(-1, pred_logits.shape[-1])
            target_flat = batch_targets.view(-1)
            
            all_predictions.append(pred_flat)
            all_targets.append(target_flat)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    results = evaluate_classification_performance(
        all_predictions, all_targets, line_type, model_name
    )
    
    return results

def evaluate_with_dummy_data():
    """
    Example evaluation with dummy data (for testing)
    """
    print("Running evaluation with dummy data...")
    
    # Create dummy data
    torch.manual_seed(42)
    num_samples = 1000
    
    predictions = torch.randn(num_samples, 4)
    
    # Generate targets with realistic class distribution
    targets = torch.cat([
        torch.zeros(600),  # 60% no_node
        torch.ones(200),   # 20% end_node
        torch.full((150,), 2),  # 15% intermediate_node
        torch.full((50,), 3)   # 5% intersection_node
    ]).long()
    
    perm = torch.randperm(num_samples)
    targets = targets[perm]
    
    results = evaluate_classification_performance(
        predictions, targets, "railroads", "LDTR_Model"
    )
    
    print(f"Evaluation complete! Results saved to eval_railroads.txt")
    print(f"Overall accuracy: {results['multi_class']['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_with_dummy_data()
    
    print("\nKey Metrics:")
    print(f"Multi-class Accuracy: {results['multi_class']['accuracy']:.4f}")
    print(f"Binary LDTR Accuracy: {results['binary_ldtr']['accuracy']:.4f}")
    print(f"Binary NTC Accuracy: {results['binary_ntc']['accuracy']:.4f}")
    
    print("\nPer-class F1 scores:")
    for i, class_name in enumerate(['no_node', 'end_node', 'intermediate_node', 'intersection_node']):
        f1 = results['multi_class']['per_class']['f1'][i]
        print(f"  {class_name}: {f1:.4f}")
    
    print("\nClasses being missed:")
    for class_name, metrics in results['analysis']['class_analysis'].items():
        if metrics['is_missed']:
            print(f"  {class_name}: recall = {metrics['recall']:.4f}")
