#!/bin/bash
# setup.sh - Environment setup script for NER fine-tuning

echo "🚀 Setting up NER Fine-tuning Environment"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create directory structure
echo "📁 Creating directories..."
mkdir -p data models outputs configs

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv ner_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ner_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify GPU availability
echo "🖥️  Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('No GPU detected')"

# Setup W&B (optional)
echo "📊 Setting up Weights & Biases..."
echo "Run 'wandb login' to authenticate with your W&B account (optional but recommended)"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x *.py

# Create example config file
echo "⚙️  Creating example configuration..."
cat > configs/example_config.json << 'EOF'
{
    "model_name": "microsoft/Phi-3.5-mini-instruct",
    "max_length": 512,
    "dataset_paths": {
        "train": "data/train_ner.jsonl",
        "val": "data/val_ner.jsonl",
        "test": "data/test_ner.jsonl"
    },
    "output_dir": "./models/ner_finetuned_model",
    "training_args": {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 500,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": true,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": false,
        "fp16": true,
        "gradient_checkpointing": true,
        "dataloader_pin_memory": false,
        "remove_unused_columns": false,
        "report_to": "wandb"
    },
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "generation_config": {
        "max_new_tokens": 150,
        "temperature": 0.1,
        "do_sample": false,
        "pad_token_id": null
    }
}
EOF

echo "✅ Setup Complete!"

echo ""
echo "📋 Next Steps:"
echo "1. Activate environment: source ner_env/bin/activate"
echo "2. Login to W&B (optional): wandb login"
echo "3. Place your CSV dataset in the project folder"
echo "4. Run preprocessing: python preprocess_data.py --csv_path your_dataset.csv"
echo "5. Start training: python train_model.py --model_name microsoft/Phi-3.5-mini-instruct"
echo "6. Evaluate results: python evaluate_model.py --model_path models/ner_finetuned_model"
echo ""
echo "🔥 Quick Start (full pipeline):"
echo "python run_pipeline.py --csv_path your_dataset.csv --model_name microsoft/Phi-3.5-mini-instruct"
echo ""
echo "📊 Files created:"
echo "  - Virtual environment: ner_env/"
echo "  - Directory structure: data/, models/, outputs/, configs/"
echo "  - Example config: configs/example_config.json"
echo ""
echo "🎯 Recommended SLMs:"
echo "  - microsoft/Phi-3.5-mini-instruct (3.8B - best balance)"
echo "  - HuggingFaceTB/SmolLM-1.7B-Instruct (1.7B - most efficient)"
echo "  - Qwen/Qwen2-1.5B-Instruct (1.5B - good multilingual)"
echo "  - google/gemma-2-2b-it (2B - strong reasoning)"