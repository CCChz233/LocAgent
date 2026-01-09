#!/usr/bin/env python3
"""
Download CodeRankEmbed model from Hugging Face Hub
"""
from transformers import AutoTokenizer, AutoModel
import os

def download_model():
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"  # CodeRankEmbed base model
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "coderankembed")
    
    print(f"Downloading {model_name} to {save_dir}...")
    
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Save locally
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    
    print(f"✓ Model successfully downloaded to {save_dir}")

if __name__ == "__main__":
    download_model()
