#!/usr/bin/env python3
"""
SLM Fine-tuning for Named Entity Recognition (NER)
Based on the smol-course methodology with LoRA/QLoRA for parameter-efficient fine-tuning
"""

import os
import json
import torch
import wandb
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from accelerate import Accelerator
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings
import bitsandbytes as bnb
import torch

torch.cuda.empty_cache()


warnings.filterwarnings('ignore')

class NERSLMTrainer:
    """
    Specialized trainer for fine-tuning Small Language Models on NER tasks
    """
    
    def __init__(self, config_path: str = None):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.accelerator = Accelerator()
        self.model = None
        self.tokenizer = None
        self.datasets = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'model_name': 'microsoft/Phi-3.5-mini-instruct',
            'max_length': 512,
            'dataset_paths': {
                'train': 'data/train_ner.jsonl',
                'val': 'data/val_ner.jsonl', 
                'test': 'data/test_ner.jsonl'
            },
            'output_dir': './models/ner_finetuned_model',
            'training_args': {
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4,
                'learning_rate': 2e-5,
                'num_train_epochs': 3,
                'warmup_ratio': 0.1,
                'logging_steps': 10,
                'eval_steps': 100,
                'save_steps': 500,
                'evaluation_strategy': 'steps',
                'save_strategy': 'steps',
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'fp16': True,
                'gradient_checkpointing': True,
                'dataloader_pin_memory': False,
                'remove_unused_columns': False,
                'report_to': 'wandb'
            },
            'lora_config': {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            },
            'generation_config': {
                'max_new_tokens': 150,
                'temperature': 0.1,
                'do_sample': False,
                'pad_token_id': None  # Will be set dynamically
            }
        }
    
    def load_datasets(self):
        """Load and prepare datasets"""
        print("Loading datasets...")
        
        datasets = {}
        for split, path in self.config['dataset_paths'].items():
            if os.path.exists(path):
                data = []
                with open(path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                datasets[split] = Dataset.from_list(data)
                print(f"Loaded {len(data)} samples for {split}")
        
        self.datasets = DatasetDict(datasets)
        return self.datasets
    
    def load_model_and_tokenizer(self):
        """Load and prepare model and tokenizer"""
        print(f"Loading model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Update generation config
        if 'generation_config' not in self.config:
            self.config['generation_config'] = {}
        self.config['generation_config']['pad_token_id'] = self.tokenizer.pad_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16 if self.config['training_args']['fp16'] else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        self.model.gradient_checkpointing_enable()
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        if 'lora_config' not in self.config:
            self.config['lora_config'] = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
            }
        lora_config = LoraConfig(**self.config['lora_config'])

        self.model = get_peft_model(self.model, lora_config)
        
        # Print model info
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def preprocess_function(self, examples):
        """Preprocess examples for training"""
        inputs = []
        
        for messages in examples['messages']:
            # Convert to chat format
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            inputs.append(chat_text)
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['max_length'],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    def prepare_datasets(self):
        """Tokenize and prepare datasets"""
        print("Tokenizing datasets...")
        
        tokenized_datasets = self.datasets.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets['train'].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_datasets
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple accuracy metric (you can enhance this with NER-specific metrics)
        accuracy = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
        
        return {
            'accuracy': accuracy,
            'prediction_samples': decoded_preds[:3],  # Log first 3 predictions
            'label_samples': decoded_labels[:3]
        }
    
    def evaluate_model(self, eval_dataset):
        """Comprehensive model evaluation"""
        print("Running evaluation...")
        
        self.model.eval()
        predictions = []
        references = []
        
        for i, example in enumerate(eval_dataset):
            if i >= 100:  # Limit evaluation samples for speed
                break
                
            # Get user message
            messages = example['messages']
            user_msg = messages[1]['content']  # User message
            expected_response = messages[2]['content']  # Assistant response
            
            # Generate prediction
            inputs = self.tokenizer.apply_chat_template(
                messages[:2],  # System + user messages only
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **self.config['generation_config'],
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            prediction = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            ).strip()
            
            predictions.append(prediction)
            references.append(expected_response)
        
        # Calculate metrics
        exact_matches = sum([pred == ref for pred, ref in zip(predictions, references)])
        accuracy = exact_matches / len(predictions)
        
        print(f"Evaluation Results:")
        print(f"  Samples evaluated: {len(predictions)}")
        print(f"  Exact match accuracy: {accuracy:.4f}")
        
        # Log sample predictions
        print(f"\nSample Predictions:")
        for i in range(min(3, len(predictions))):
            print(f"  Example {i+1}:")
            print(f"    Predicted: {predictions[i]}")
            print(f"    Expected:  {references[i]}")
            print()
        
        return {
            'accuracy': accuracy,
            'exact_matches': exact_matches,
            'total_samples': len(predictions)
        }
    
    def train(self):
        """Main training function"""
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="ner-slm-finetuning",
            name=f"ner_{self.config['model_name'].split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config
        )
        
        # Load data and model
        self.load_datasets()
        self.load_model_and_tokenizer()
        tokenized_datasets = self.prepare_datasets()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            **self.config['training_args']
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'] if 'val' in tokenized_datasets else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        trainer.save_state()
        
        # Evaluate
        if 'test' in tokenized_datasets:
            eval_results = self.evaluate_model(self.datasets['test'])
            wandb.log(eval_results)
        
        # Log training results
        wandb.log(train_result.metrics)
        
        print("Training completed!")
        print(f"Model saved to: {self.config['output_dir']}")
        
        return trainer, train_result

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Fine-tune SLM for NER")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./models/ner_finetuned_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Create config if not provided
    if not args.config:
        config = {
            'model_name': args.model_name,
            'output_dir': args.output_dir,
            'dataset_paths': {
                'train': 'data/train_ner.jsonl',
                'val': 'data/val_ner.jsonl',
                'test': 'data/test_ner.jsonl'
            },
            'max_length': 512,
            'training_args': {
                'per_device_train_batch_size': 1,          # Reduce batch size
                'per_device_eval_batch_size': 1,
                'gradient_accumulation_steps': 4,          # Accumulate gradients to simulate larger batch
                'learning_rate': 2e-5,
                'num_train_epochs': 3,
                #'max_length': 256,                          # Limit max tokens to 256 (adjust if needed)
                'warmup_ratio': 0.1,
                'logging_steps': 10,
                'eval_steps': 100,
                'save_steps': 500,
                'evaluation_strategy': 'steps',
                'save_strategy': 'steps',
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'fp16': True,                               # Enable mixed precision
                'gradient_checkpointing': True,             # Enable gradient checkpointing
                'dataloader_pin_memory': False,
                'remove_unused_columns': False,
                'report_to': 'wandb',
                "deepspeed": "deepspeed_config.json"
            }
        }
        
        os.makedirs('configs', exist_ok=True)
        with open('configs/auto_training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        config_path = 'configs/auto_training_config.json'
        print(f"Created config file: {config_path}")
    else:
        config_path = args.config
    
    print(f"Starting training with model: {args.model_name}")
    trainer = NERSLMTrainer(config_path)
    model, results = trainer.train()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()