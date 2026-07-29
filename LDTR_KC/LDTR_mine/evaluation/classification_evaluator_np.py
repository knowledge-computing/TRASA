#!/usr/bin/env python3

import numpy as np
import os
import argparse
from datetime import datetime

class ClassificationEvaluator:
  
    def __init__(self, line_type="railroads"):
        self.line_type = line_type
        self.class_names = ['no_node', 'end_node', 'intermediate_node', 'intersection_node']
        self.output_file = f"eval_{line_type}.txt"
        
    def evaluate_multi_class(self, predictions, targets, model_name="Model"):

        if predictions.ndim > 1:
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            pred_classes = predictions
            
        accuracy = self._calculate_accuracy(targets, pred_classes)
        precision, recall, f1, support = self._calculate_precision_recall_f1(targets, pred_classes)
        
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        total_samples = np.sum(support)
        precision_weighted = np.sum(precision * support) / total_samples
        recall_weighted = np.sum(recall * support) / total_samples
        f1_weighted = np.sum(f1 * support) / total_samples
        
        cm = self._calculate_confusion_matrix(targets, pred_classes)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            },
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names
        }
        
        return results
    
    def evaluate_binary_classification(self, predictions, targets, model_name="Model", 
                                    binary_type="LDTR"):

        if predictions.ndim > 1:
            pred_classes = np.argmax(predictions, axis=-1)
        else:
            pred_classes = predictions
            
        if binary_type == "LDTR":
            binary_pred = (pred_classes > 0).astype(int)
            binary_target = (targets > 0).astype(int)
        elif binary_type == "NTC":
            binary_pred = (pred_classes > 0).astype(int)
            binary_target = (targets > 0).astype(int)
        else:
            raise ValueError("binary_type must be 'LDTR' or 'NTC'")
        
        accuracy = self._calculate_accuracy(binary_target, binary_pred)
        precision, recall, f1, support = self._calculate_precision_recall_f1(binary_target, binary_pred)
        
        cm = self._calculate_confusion_matrix(binary_target, binary_pred)
        
        results = {
            'model_name': model_name,
            'binary_type': binary_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'confusion_matrix': cm.tolist(),
            'class_names': ['no_node', 'any_node']
        }
        
        return results
    
    def _calculate_accuracy(self, targets, predictions):
        return np.mean(targets == predictions)
    
    def _calculate_precision_recall_f1(self, targets, predictions):
        num_classes = len(self.class_names)
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        support = np.zeros(num_classes)
        
        for i in range(num_classes):
            tp = np.sum((predictions == i) & (targets == i))
            fp = np.sum((predictions == i) & (targets != i))
            fn = np.sum((predictions != i) & (targets == i))
            
            support[i] = tp + fn
            
            if tp + fp > 0:
                precision[i] = tp / (tp + fp)
            else:
                precision[i] = 0.0
            
            if tp + fn > 0:
                recall[i] = tp / (tp + fn)
            else:
                recall[i] = 0.0
            
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1[i] = 0.0
        
        return precision, recall, f1, support
    
    def _calculate_confusion_matrix(self, targets, predictions):
        num_classes = len(self.class_names)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for i in range(len(targets)):
            true_class = int(targets[i])
            pred_class = int(predictions[i])
            cm[true_class, pred_class] += 1
        
        return cm
    
    def analyze_class_performance(self, results):
        cm = np.array(results['confusion_matrix'])
        precision = results['per_class']['precision']
        recall = results['per_class']['recall']
        f1 = results['per_class']['f1']
        
        analysis = {
            'worst_performing_classes': [],
            'most_confused_classes': [],
            'class_analysis': {}
        }
        
        f1_scores = [(i, f1[i]) for i in range(len(f1))]
        f1_scores.sort(key=lambda x: x[1])
        analysis['worst_performing_classes'] = [
            (self.class_names[i], f1[i]) for i, _ in f1_scores[:2]
        ]
        
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((
                        self.class_names[i], 
                        self.class_names[j], 
                        cm[i, j]
                    ))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        analysis['most_confused_classes'] = confusion_pairs[:3]
        
        for i, class_name in enumerate(self.class_names):
            analysis['class_analysis'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': results['per_class']['support'][i],
                'is_missed': recall[i] < 0.5, 
                'is_confused': precision[i] < 0.5  
            }
        
        return analysis
    
    def save_results(self, multi_class_results, binary_results_ldtr, binary_results_ntc, analysis):
        
        with open(self.output_file, 'w') as f:      
            f.write("Per-Class Metrics:\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name:<20} "
                       f"{multi_class_results['per_class']['precision'][i]:<10.4f} "
                       f"{multi_class_results['per_class']['recall'][i]:<10.4f} "
                       f"{multi_class_results['per_class']['f1'][i]:<10.4f} "
                       f"{multi_class_results['per_class']['support'][i]:<10}\n")
            
            f.write("\nMacro Averages:\n")
            f.write(f"Precision: {multi_class_results['macro_avg']['precision']:.4f}\n")
            f.write(f"Recall: {multi_class_results['macro_avg']['recall']:.4f}\n")
            f.write(f"F1: {multi_class_results['macro_avg']['f1']:.4f}\n\n")
            
            f.write("Weighted Averages:\n")
            f.write(f"Precision: {multi_class_results['weighted_avg']['precision']:.4f}\n")
            f.write(f"Recall: {multi_class_results['weighted_avg']['recall']:.4f}\n")
            f.write(f"F1: {multi_class_results['weighted_avg']['f1']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"{'':<15}")
            for class_name in self.class_names:
                f.write(f"{class_name[:8]:<10}")
            f.write("\n")
            
            cm = np.array(multi_class_results['confusion_matrix'])
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name[:14]:<15}")
                for j in range(len(self.class_names)):
                    f.write(f"{cm[i, j]:<10}")
                f.write("\n")
            f.write("\n")
            
            
            f.write(f"LDTR Binary Classification:\n")
            f.write(f"Accuracy: {binary_results_ldtr['accuracy']:.4f}\n")
            f.write(f"No Node - Precision: {binary_results_ldtr['precision'][0]:.4f}, "
                   f"Recall: {binary_results_ldtr['recall'][0]:.4f}, "
                   f"F1: {binary_results_ldtr['f1'][0]:.4f}\n")
            f.write(f"Any Node - Precision: {binary_results_ldtr['precision'][1]:.4f}, "
                   f"Recall: {binary_results_ldtr['recall'][1]:.4f}, "
                   f"F1: {binary_results_ldtr['f1'][1]:.4f}\n\n")
            
            f.write(f"NTC Binary Classification:\n")
            f.write(f"Accuracy: {binary_results_ntc['accuracy']:.4f}\n")
            f.write(f"No Node - Precision: {binary_results_ntc['precision'][0]:.4f}, "
                   f"Recall: {binary_results_ntc['recall'][0]:.4f}, "
                   f"F1: {binary_results_ntc['f1'][0]:.4f}\n")
            f.write(f"Any Node - Precision: {binary_results_ntc['precision'][1]:.4f}, "
                   f"Recall: {binary_results_ntc['recall'][1]:.4f}, "
                   f"F1: {binary_results_ntc['f1'][1]:.4f}\n\n")
            
            for class_name, metrics in analysis['class_analysis'].items():
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1: {metrics['f1']:.4f}\n")
                f.write(f"    Support: {metrics['support']}\n")
                f.write(f"    Is Missed: {metrics['is_missed']}\n")
                f.write(f"    Is Confused: {metrics['is_confused']}\n")
                f.write("\n")
            


def evaluate_classification_performance(predictions, targets, line_type="railroads", model_name="Model"):

    evaluator = ClassificationEvaluator(line_type)
    
    multi_class_results = evaluator.evaluate_multi_class(predictions, targets, model_name)
    
    binary_results_ldtr = evaluator.evaluate_binary_classification(
        predictions, targets, model_name, "LDTR"
    )
    binary_results_ntc = evaluator.evaluate_binary_classification(
        predictions, targets, model_name, "NTC"
    )
    
    analysis = evaluator.analyze_class_performance(multi_class_results)
    
    evaluator.save_results(multi_class_results, binary_results_ldtr, binary_results_ntc, analysis)
    
    return {
        'multi_class': multi_class_results,
        'binary_ldtr': binary_results_ldtr,
        'binary_ntc': binary_results_ntc,
        'analysis': analysis
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate classification performance')
    parser.add_argument('--line_type', type=str, default='railroads', 
                       help='Type of line being evaluated')
    parser.add_argument('--model_name', type=str, default='LDTR_Model', 
                       help='Name of the model being evaluated')
    
    args = parser.parse_args()
    
    print(f"Generating example evaluation for {args.line_type}...")
    
    np.random.seed(42)
    num_samples = 1000
    
    predictions = np.random.randn(num_samples, 4)
    
    targets = np.concatenate([
        np.zeros(600),  # 60% no_node
        np.ones(200),   # 20% end_node
        np.full(150, 2),  # 15% intermediate_node
        np.full(50, 3)   # 5% intersection_node
    ]).astype(int)
    
    perm = np.random.permutation(num_samples)
    targets = targets[perm]
    
    results = evaluate_classification_performance(
        predictions, targets, args.line_type, args.model_name
    )
    
    print(f"rResults saved to eval_{args.line_type}.txt")