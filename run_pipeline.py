#!/usr/bin/env python3
"""
Complete pipeline runner - preprocessing, training, and evaluation
Usage: python run_pipeline.py --csv_path /Users/alexanderwoods/SLM-FineTuning-For-NER/generated_sample.csv --model_name microsoft/Phi-3.5-mini-instruct
"""

import argparse
import os
import subprocess
import sys
import json

def run_command(command):
    """Run a command and handle errors"""
    print(f"\n🔄 Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running command: {' '.join(command)}")
        print(f"Error output: {result.stderr}")
        print(f"Standard output: {result.stdout}")
        sys.exit(1)
    
    print(f"✅ Success: {' '.join(command)}")
    if result.stdout.strip():
        print(f"Output: {result.stdout}")
    return result

def check_file_exists(file_path, description):
    """Check if a file exists and raise error if not"""
    if not os.path.exists(file_path):
        print(f"❌ {description} not found: {file_path}")
        sys.exit(1)
    print(f"✅ Found {description}: {file_path}")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'outputs', 'configs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}/")

def main():
    parser = argparse.ArgumentParser(description="Run complete NER fine-tuning pipeline")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct", help="Model to fine-tune")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    print("🚀 Starting NER Fine-tuning Pipeline")
    print(f"📊 Dataset: {args.csv_path}")
    print(f"🤖 Model: {args.model_name}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.learning_rate}")
    
    # Create directory structure
    create_directories()
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n" + "="*50)
        print("📊 Step 1: Data Preprocessing")
        print("="*50)
        
        check_file_exists(args.csv_path, "CSV dataset")
        
        run_command([
            sys.executable, "preprocess_data.py", 
            "--csv_path", args.csv_path
        ])
        
        # Verify preprocessing outputs
        required_files = [
            "data/train_ner.jsonl",
            "data/val_ner.jsonl", 
            "data/test_ner.jsonl"
        ]
        
        for file_path in required_files:
            check_file_exists(file_path, f"Preprocessed data file")
    
    # Step 2: Model Training  
    if not args.skip_training:
        print("\n" + "="*50)
        print("🤖 Step 2: Model Training")
        print("="*50)
        
        # Check if preprocessed data exists
        check_file_exists("data/train_ner.jsonl", "Training data")
        
        run_command([
            sys.executable, "train_model.py",
            "--model_name", args.model_name,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--learning_rate", str(args.learning_rate)
        ])
        
        # Verify training outputs
        model_path = "models/ner_finetuned_model"
        check_file_exists(model_path, "Trained model directory")
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        print("\n" + "="*50)
        print("📈 Step 3: Model Evaluation")
        print("="*50)
        
        model_path = "models/ner_finetuned_model"
        test_file = "data/test_ner.jsonl"
        
        check_file_exists(model_path, "Trained model")
        check_file_exists(test_file, "Test data")
        
        run_command([
            sys.executable, "evaluate_model.py",
            "--model_path", model_path,
            "--test_file", test_file,
            "--base_model", args.model_name
        ])
    
    print("\n" + "="*60)
    print("🎉 Pipeline Complete!")
    print("="*60)
    
    print("\n📋 Generated Files:")
    
    # List generated files
    file_checks = [
        ("📊 Preprocessed Data:", [
            "data/train_ner.jsonl",
            "data/val_ner.jsonl", 
            "data/test_ner.jsonl"
        ]),
        ("🤖 Model:", [
            "models/ner_finetuned_model/"
        ]),
        ("📈 Evaluation Results:", [
            "outputs/evaluation_report.txt",
            "outputs/ner_evaluation_plots.png"
        ]),
        ("⚙️  Configuration:", [
            "configs/auto_training_config.json",
            "configs/ner_training_config.json"
        ])
    ]
    
    for category, files in file_checks:
        print(f"\n{category}")
        for file_path in files:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  ⏭️  {file_path} (not generated - possibly skipped)")
    
    print(f"\n🔧 Next Steps:")
    print(f"  1. Review evaluation report: outputs/evaluation_report.txt")
    print(f"  2. Check plots: outputs/ner_evaluation_plots.png")
    print(f"  3. Use model for inference: python evaluate_model.py --model_path models/ner_finetuned_model --demo_only")

if __name__ == "__main__":
    main()