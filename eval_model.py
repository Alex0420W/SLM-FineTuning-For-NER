#!/usr/bin/env python3
"""
Evaluation and Inference script for NER fine-tuned SLMs
"""

import json
import torch
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

class NERModelEvaluator:
    """
    Comprehensive evaluation suite for NER fine-tuned models
    """
    
    def __init__(self, model_path: str, base_model_name: str = None):
        """Initialize evaluator with model path"""
        self.model_path = model_path
        self.base_model_name = base_model_name or "microsoft/Phi-3.5-mini-instruct"
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def extract_entities_from_text(self, text: str, max_new_tokens: int = 150) -> Dict:
        """
        Extract entities from a given text using the fine-tuned model
        """
        messages = [
            {
                "role": "system",
                "content": "You are a named entity recognition expert. Extract and classify named entities from text."
            },
            {
                "role": "user",
                "content": f"Extract named entities from: {text}"
            }
        ]
        
        # Format for the model
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        ).strip()
        
        # Try to parse JSON response
        try:
            entities = json.loads(response)
            if isinstance(entities, dict):
                return entities
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract entities using regex patterns
        return self._extract_entities_fallback(response)
    
    def _extract_entities_fallback(self, response: str) -> Dict:
        """Fallback method to extract entities from response text"""
        entities = {}
        
        # Common patterns for entity extraction
        patterns = {
            'PERSON': r'PERSON[:\s]*([^,\n]+)',
            'LOC': r'LOC[ATION]*[:\s]*([^,\n]+)', 
            'ORG': r'ORG[ANIZATION]*[:\s]*([^,\n]+)',
            'MISC': r'MISC[ELLANEOUS]*[:\s]*([^,\n]+)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                entities[entity_type] = [match.strip() for match in matches]
        
        return entities
    
    def evaluate_on_dataset(self, test_file: str) -> Dict:
        """
        Comprehensive evaluation on test dataset
        """
        print(f"Evaluating on {test_file}")
        
        # Load test data
        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        results = {
            'predictions': [],
            'ground_truth': [],
            'exact_matches': 0,
            'partial_matches': 0,
            'entity_level_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        }
        
        print(f"Processing {len(test_data)} samples...")
        
        for i, sample in enumerate(test_data):
            if i % 50 == 0:
                print(f"Processed {i}/{len(test_data)} samples")
            
            # Extract text from user message
            user_msg = sample['messages'][1]['content']
            text = user_msg.replace("Extract named entities from: ", "")
            
            # Get ground truth
            try:
                ground_truth = json.loads(sample['messages'][2]['content'])
            except json.JSONDecodeError:
                continue
            
            # Get prediction
            predicted = self.extract_entities_from_text(text)
            
            results['predictions'].append(predicted)
            results['ground_truth'].append(ground_truth)
            
            # Calculate matches
            exact_match = self._exact_match(predicted, ground_truth)
            partial_match = self._partial_match(predicted, ground_truth)
            
            if exact_match:
                results['exact_matches'] += 1
            if partial_match:
                results['partial_matches'] += 1
            
            # Entity-level metrics
            self._update_entity_metrics(predicted, ground_truth, results['entity_level_metrics'])
        
        # Calculate final metrics
        total_samples = len(results['predictions'])
        results['exact_match_accuracy'] = results['exact_matches'] / total_samples
        results['partial_match_accuracy'] = results['partial_matches'] / total_samples
        
        # Calculate precision, recall, F1 for each entity type
        results['entity_metrics'] = {}
        for entity_type, metrics in results['entity_level_metrics'].items():
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['entity_metrics'][entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': metrics['tp'] + metrics['fn']
            }
        
        return results
    
    def _exact_match(self, pred: Dict, truth: Dict) -> bool:
        """Check if prediction exactly matches ground truth"""
        if set(pred.keys()) != set(truth.keys()):
            return False
        
        for entity_type in pred:
            pred_set = set(pred[entity_type]) if entity_type in pred else set()
            truth_set = set(truth[entity_type]) if entity_type in truth else set()
            if pred_set != truth_set:
                return False
        
        return True
    
    def _partial_match(self, pred: Dict, truth: Dict) -> bool:
        """Check if prediction has any overlap with ground truth"""
        for entity_type in set(pred.keys()).union(set(truth.keys())):
            pred_set = set(pred.get(entity_type, []))
            truth_set = set(truth.get(entity_type, []))
            if pred_set.intersection(truth_set):
                return True
        return False
    
    def _update_entity_metrics(self, pred: Dict, truth: Dict, metrics: Dict):
        """Update entity-level metrics"""
        all_entity_types = set(pred.keys()).union(set(truth.keys()))
        
        for entity_type in all_entity_types:
            pred_entities = set(pred.get(entity_type, []))
            truth_entities = set(truth.get(entity_type, []))
            
            tp = len(pred_entities.intersection(truth_entities))
            fp = len(pred_entities - truth_entities)
            fn = len(truth_entities - pred_entities)
            
            metrics[entity_type]['tp'] += tp
            metrics[entity_type]['fp'] += fp
            metrics[entity_type]['fn'] += fn
    
    def generate_report(self, results: Dict, output_file: str = "outputs/evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        report = f"""
NER Model Evaluation Report
==========================

Overall Performance:
- Total samples: {len(results['predictions'])}
- Exact Match Accuracy: {results['exact_match_accuracy']:.4f}
- Partial Match Accuracy: {results['partial_match_accuracy']:.4f}

Entity-Level Metrics:
"""
        
        for entity_type, metrics in results['entity_metrics'].items():
            report += f"""
{entity_type}:
  Precision: {metrics['precision']:.4f}
  Recall:    {metrics['recall']:.4f}
  F1-Score:  {metrics['f1']:.4f}
  Support:   {metrics['support']}
"""
        
        # Calculate macro averages
        if results['entity_metrics']:
            precisions = [m['precision'] for m in results['entity_metrics'].values()]
            recalls = [m['recall'] for m in results['entity_metrics'].values()]
            f1s = [m['f1'] for m in results['entity_metrics'].values()]
            
            report += f"""
Macro Averages:
  Precision: {np.mean(precisions):.4f}
  Recall:    {np.mean(recalls):.4f}
  F1-Score:  {np.mean(f1s):.4f}
"""
        
        report += "\nSample Predictions:\n"
        
        # Add sample predictions
        for i in range(min(5, len(results['predictions']))):
            report += f"""
Example {i+1}:
  Predicted: {results['predictions'][i]}
  Ground Truth: {results['ground_truth'][i]}
  Match: {'✓' if self._exact_match(results['predictions'][i], results['ground_truth'][i]) else '✗'}
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")
        print(report)
        
        return report
    
    def visualize_results(self, results: Dict, save_path: str = "outputs/ner_evaluation_plots.png"):
        """Create visualization of evaluation results"""
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        if not results['entity_metrics']:
            print("No entity metrics to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall accuracy
        accuracies = [
            results['exact_match_accuracy'],
            results['partial_match_accuracy']
        ]
        ax1.bar(['Exact Match', 'Partial Match'], accuracies, color=['#2E8B57', '#4682B4'])
        ax1.set_title('Overall Accuracy Metrics')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # 2. Entity-wise F1 scores
        entity_types = list(results['entity_metrics'].keys())
        f1_scores = [results['entity_metrics'][et]['f1'] for et in entity_types]
        
        ax2.bar(entity_types, f1_scores, color='#FF6347')
        ax2.set_title('F1-Score by Entity Type')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Precision vs Recall
        precisions = [results['entity_metrics'][et]['precision'] for et in entity_types]
        recalls = [results['entity_metrics'][et]['recall'] for et in entity_types]
        
        ax3.scatter(precisions, recalls, s=100, alpha=0.7, c=range(len(entity_types)), cmap='viridis')
        for i, et in enumerate(entity_types):
            ax3.annotate(et, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        ax3.set_xlabel('Precision')
        ax3.set_ylabel('Recall')
        ax3.set_title('Precision vs Recall by Entity Type')
        ax3.grid(True, alpha=0.3)
        
        # 4. Support (number of true entities) by type
        supports = [results['entity_metrics'][et]['support'] for et in entity_types]
        ax4.bar(entity_types, supports, color='#32CD32')
        ax4.set_title('Number of True Entities by Type')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to {save_path}")

class NERInference:
    """
    Simple inference class for using the trained NER model
    """
    
    def __init__(self, model_path: str, base_model_name: str = None):
        """Initialize inference engine"""
        self.evaluator = NERModelEvaluator(model_path, base_model_name)
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities from text"""
        return self.evaluator.extract_entities_from_text(text)
    
    def batch_extract(self, texts: List[str]) -> List[Dict]:
        """Extract entities from multiple texts"""
        results = []
        for text in texts:
            results.append(self.extract_entities(text))
        return results
    
    def format_output(self, entities: Dict, format_type: str = "pretty") -> str:
        """Format extracted entities for display"""
        
        if format_type == "json":
            return json.dumps(entities, indent=2)
        
        elif format_type == "pretty":
            if not entities:
                return "No entities found."
            
            output = "Extracted Entities:\n"
            for entity_type, entity_list in entities.items():
                if entity_list:
                    output += f"\n{entity_type}:\n"
                    for entity in entity_list:
                        output += f"  • {entity}\n"
            return output
        
        elif format_type == "table":
            data = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    data.append({"Entity Type": entity_type, "Entity": entity})
            
            if data:
                df = pd.DataFrame(data)
                return df.to_string(index=False)
            else:
                return "No entities found."
        
        return str(entities)

def main():
    """Main function for evaluation and inference"""
    
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_file", type=str, default="data/test_ner.jsonl", help="Test dataset file")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Base model name")
    parser.add_argument("--demo_only", action="store_true", help="Run demo inference only")
    
    args = parser.parse_args()
    
    print(f"Evaluating model: {args.model_path}")
    
    if not args.demo_only:
        # Full evaluation
        evaluator = NERModelEvaluator(args.model_path, args.base_model)
        
        if os.path.exists(args.test_file):
            results = evaluator.evaluate_on_dataset(args.test_file)
            evaluator.generate_report(results)
            evaluator.visualize_results(results)
        else:
            print(f"Test file not found: {args.test_file}")
    
    # Demo inference
    print("\n=== Demo Inference ===")
    inference = NERInference(args.model_path, args.base_model)
    
    demo_texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "The European Union headquarters is located in Brussels, Belgium.",
        "Microsoft CEO Satya Nadella announced new AI initiatives at the conference in Seattle.",
        "The FIFA World Cup 2022 took place in Qatar from November to December."
    ]
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\nExample {i}: {text}")
        entities = inference.extract_entities(text)
        print(inference.format_output(entities, "pretty"))
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()