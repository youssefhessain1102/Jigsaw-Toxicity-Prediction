import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import ToxicityDataset
from model import DistilBERTClassifier
from trainer import Trainer
from preprocessor import load_and_preprocess, save_cleaned

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import yaml
import random
import tempfile

def load_config() -> dict:
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def seed_setting(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_and_save(df: pd.DataFrame, config: dict) -> tuple[str, str, str]:
    seed = config['training']['seed']
    val_split = config['training']['val_split']

    train_and_val, test = train_test_split(df, random_state=seed, shuffle=True, test_size=val_split)

    val_split = val_split / (1 - val_split)
    train, val = train_test_split(train_and_val, random_state=seed, shuffle=True, test_size=val_split)
    print(f"Split sizes → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    def save_temp(data: pd.DataFrame) -> str:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        data.to_csv(temp.name, index=False)
        return temp.name
    
    return save_temp(train), save_temp(val), save_temp(test)

def evaluate_test(model, test_loader, device, threshold: float):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
        
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        binary_preds = (all_preds >= threshold).astype(int)

        f1_Score = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
        auc_roc_score = roc_auc_score(all_labels, binary_preds, average='macro')

        print("\n── Test Set Results ──────────────────────")
        print(f"  F1  (macro): {f1_Score:.4f}")
        print(f"  AUC (macro): {auc_roc_score:.4f}")
        print("──────────────────────────────────────────")

def main():
    # Loading Config file(yaml)
    config = load_config()
    seed_setting(config['training']['seed'])

    # Data Loading and Cleaning
    data = load_and_preprocess(config)
    cleaned_df = save_cleaned(data, config)
    df = pd.read_csv(cleaned_df)

    # Data Splitting into Train & Test & Validation
    train_path, test_path, val_path = split_and_save(df, config)

    # Tokenizer & Model
    tokenizer = DistilBertTokenizer.from_pretrained(config['model']['name'])
    model = DistilBERTClassifier(config['model']['name'], dropout=config['model']['dropout'], num_labels=config['model']['num_labels'])

    # Datasets of Train & Test & Validation
    max_length = config['model']['max_length']
    train_dataset = ToxicityDataset(
        context_length=max_length,
        df=train_path,
        tokenizer=tokenizer
    )
    test_dataset = ToxicityDataset(
        context_length=max_length,
        df=test_path,
        tokenizer=tokenizer
    )
    val_dataset = ToxicityDataset(
        context_length=max_length,
        df=val_path,
        tokenizer=tokenizer
    )

    # Dataloaders of Train & Test & Validation
    loader_kwargs = {
        "batch_size": config['training']['batch_size'],
        "num_workers": config['training']['dataloader_num_workers'],
        "pin_memory": config['training']['pin_memory'],
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Model Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    trainer.train()

    # Saving best Model
    best_model_path = os.path.join(
        config['paths']['model_save_dir'], 
        config['paths']['best_model_name']
    )
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Model evaluation with f1_score & roc_auc_score
    evaluate_test(model, test_loader=test_loader, device=device, threshold=config['api']['threshold'])
    
    for i in [train_path, test_path, val_path]:
        os.remove(i)
    print("\nTemp file cleaned up succesfully")

if __name__ == "__main__":
    print("started")
    main()
