#!/usr/bin/env python3
"""
NER Dataset Preprocessing Script
Converts CSV dataset to training-ready formats for SLM fine-tuning
"""
import json
import pandas as pd
import ast
from typing import List, Dict, Tuple, Any
import re
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

class NERDatasetProcessor:
    """
    A comprehensive processor for NER datasets, specifically designed for SLM fine-tuning.
    Converts CSV format to various training formats (BIO, instruction-based, etc.)
    """
    
    def __init__(self, csv_path: str):
        """Initialize with CSV file path"""
        self.csv_path = csv_path
        self.df = None
        self.processed_data = []
        self.entity_stats = {}
        self.load_data()
    
    def load_data(self):
        """Load and initial processing of CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.df)} samples from {self.csv_path}")
            self.analyze_dataset()
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def analyze_dataset(self):
        """Analyze dataset characteristics"""
        print("\n=== Dataset Analysis ===")
        print(f"Total samples: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Analyze entity distribution
        all_entities = []
        for _, row in self.df.iterrows():
            entities = ast.literal_eval(row.iloc[2])  # entities column
            for entity_type, entity_list in entities.items():
                all_entities.extend([(entity_type, entity) for entity in entity_list])
        
        entity_type_counts = Counter([e[0] for e in all_entities])
        self.entity_stats = dict(entity_type_counts)
        
        print(f"\nEntity type distribution:")
        for entity_type, count in entity_type_counts.most_common():
            print(f"  {entity_type}: {count}")
        
        print(f"\nTotal unique entities: {len(set(all_entities))}")
        print(f"Average entities per sample: {len(all_entities) / len(self.df):.2f}")
    
    def extract_text_and_entities(self, row) -> Tuple[str, Dict]:
        """Extract clean text and entities from a row"""
        # Extract text (remove quotes and extra formatting)
        text_raw = row.iloc[1]
        # Remove the "# Text" header and quotes
        text = re.sub(r'^.*?# Text\s*"*', '', text_raw).strip(' "')
        
        # Parse entities
        entities = ast.literal_eval(row.iloc[2])
        
        return text, entities
    
    def convert_to_bio_format(self) -> List[Dict]:
        """
        Convert to standard BIO format for token classification
        Returns list of samples with tokens and BIO tags
        """
        bio_data = []
        
        for idx, row in self.df.iterrows():
            text, entities = self.extract_text_and_entities(row)
            
            # Tokenize (simple whitespace tokenization - you might want to use a proper tokenizer)
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Apply entity labels
            for entity_type, entity_words in entities.items():
                for entity_phrase in entity_words:
                    entity_tokens = entity_phrase.split()
                    
                    # Find entity in text
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            # Apply BIO tagging
                            if len(entity_tokens) == 1:
                                labels[i] = entity_type
                            else:
                                labels[i] = entity_type.replace('B-', 'B-').replace('I-', 'B-')
                                for j in range(1, len(entity_tokens)):
                                    labels[i+j] = entity_type.replace('B-', 'I-')
                            break
            
            bio_data.append({
                'id': idx,
                'tokens': tokens,
                'ner_tags': labels,
                'text': text,
                'source': row.iloc[3] if len(row) > 3 else 'unknown'
            })
        
        return bio_data
    
    def convert_to_instruction_format(self) -> List[Dict]:
        """
        Convert to instruction-following format for SLM fine-tuning
        This format works well with models like Phi, Gemma, etc.
        """
        instruction_data = []
        
        system_prompt = """You are an expert at named entity recognition. Extract all named entities from the given text and classify them into the following categories:
- PERSON: Names of people
- LOCATION: Names of places, countries, cities, etc.
- ORGANIZATION: Names of companies, institutions, etc.
- MISCELLANEOUS: Other named entities

Format your response as JSON with entity types as keys and lists of entities as values."""
        
        for idx, row in self.df.iterrows():
            text, entities = self.extract_text_and_entities(row)
            
            # Clean entity types for output (remove B-/I- prefixes)
            clean_entities = {}
            for entity_type, entity_list in entities.items():
                clean_type = entity_type.replace('B-', '').replace('I-', '')
                if clean_type not in clean_entities:
                    clean_entities[clean_type] = []
                clean_entities[clean_type].extend(entity_list)
            
            # Remove duplicates
            for key in clean_entities:
                clean_entities[key] = list(set(clean_entities[key]))
            
            instruction_data.append({
                'id': idx,
                'system': system_prompt,
                'user': f"Extract named entities from this text: {text}",
                'assistant': json.dumps(clean_entities),
                'source': row.iloc[3] if len(row) > 3 else 'unknown'
            })
        
        return instruction_data
    
    def create_chat_format(self) -> List[Dict]:
        """
        Create chat format compatible with most modern SLMs
        """
        chat_data = []
        
        for idx, row in self.df.iterrows():
            text, entities = self.extract_text_and_entities(row)
            
            # Clean entities
            clean_entities = {}
            for entity_type, entity_list in entities.items():
                clean_type = entity_type.replace('B-', '').replace('I-', '')
                if clean_type not in clean_entities:
                    clean_entities[clean_type] = []
                clean_entities[clean_type].extend(entity_list)
            
            # Remove duplicates and empty lists
            clean_entities = {k: list(set(v)) for k, v in clean_entities.items() if v}
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a named entity recognition expert. Extract and classify named entities from text."
                },
                {
                    "role": "user", 
                    "content": f"Extract named entities from: {text}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(clean_entities)
                }
            ]
            
            chat_data.append({
                'id': idx,
                'messages': messages,
                'source': row.iloc[3] if len(row) > 3 else 'unknown'
            })
        
        return chat_data
    
    def split_dataset(self, data: List[Dict], test_size: float = 0.2, val_size: float = 0.1):
        """Split dataset into train/val/test"""
        # First split: train+val vs test
        train_val, test = train_test_split(data, test_size=test_size, random_state=42)
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train)} samples")
        print(f"  Validation: {len(val)} samples") 
        print(f"  Test: {len(test)} samples")
        
        return train, val, test
    
    def save_processed_data(self, data: List[Dict], output_path: str, format_type: str):
        """Save processed data in various formats"""
        
        if format_type == 'bio':
            # Save in CONLL format
            with open(f"{output_path}_bio.conll", 'w') as f:
                for sample in data:
                    for token, label in zip(sample['tokens'], sample['ner_tags']):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")
        
        elif format_type == 'jsonl':
            # Save as JSONL for modern training frameworks
            with open(f"{output_path}.jsonl", 'w') as f:
                for sample in data:
                    f.write(json.dumps(sample) + "\n")
        
        elif format_type == 'huggingface':
            # Save in HuggingFace datasets format
            with open(f"{output_path}_hf.json", 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_path} in {format_type} format")
    
    def generate_model_config(self) -> Dict:
        """Generate configuration for model training"""
        config = {
            'dataset_info': {
                'total_samples': len(self.df),
                'entity_types': list(self.entity_stats.keys()),
                'entity_distribution': self.entity_stats,
                'unique_entity_types': len(self.entity_stats),
            },
            'training_config': {
                'task_type': 'named_entity_recognition',
                'model_recommendations': [
                    'microsoft/Phi-3.5-mini-instruct',
                    'HuggingFaceTB/SmolLM-1.7B-Instruct', 
                    'Qwen/Qwen2-1.5B-Instruct',
                    'google/gemma-2-2b-it'
                ],
                'training_args': {
                    'per_device_train_batch_size': 4,
                    'per_device_eval_batch_size': 4,
                    'learning_rate': 2e-5,
                    'num_train_epochs': 3,
                    'warmup_ratio': 0.1,
                    'logging_steps': 10,
                    'eval_steps': 100,
                    'save_steps': 500,
                    'fp16': True,
                    'gradient_checkpointing': True
                },
                'lora_config': {
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    'lora_dropout': 0.1,
                    'bias': 'none',
                    'task_type': 'CAUSAL_LM'
                }
            }
        }
        
        return config

def process_ner_dataset(csv_path: str):
    """Main processing function"""
    processor = NERDatasetProcessor(csv_path)
    
    # Convert to different formats
    bio_data = processor.convert_to_bio_format()
    instruction_data = processor.convert_to_instruction_format()
    chat_data = processor.create_chat_format()
    
    # Split datasets
    train_bio, val_bio, test_bio = processor.split_dataset(bio_data)
    train_chat, val_chat, test_chat = processor.split_dataset(chat_data)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Save processed data
    processor.save_processed_data(train_chat, 'data/train_ner', 'jsonl')
    processor.save_processed_data(val_chat, 'data/val_ner', 'jsonl')
    processor.save_processed_data(test_chat, 'data/test_ner', 'jsonl')
    
    # Generate config
    config = processor.generate_model_config()
    with open('configs/ner_training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n=== Processing Complete ===")
    print("Generated files:")
    print("- data/train_ner.jsonl, data/val_ner.jsonl, data/test_ner.jsonl")
    print("- configs/ner_training_config.json")
    
    return processor, train_chat, val_chat, test_chat

def main():
    parser = argparse.ArgumentParser(description="Preprocess NER dataset for training")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    print(f"Processing dataset: {args.csv_path}")
    processor, train_data, val_data, test_data = process_ner_dataset(args.csv_path)
    print("✅ Data preprocessing complete!")

if __name__ == "__main__":
    main()