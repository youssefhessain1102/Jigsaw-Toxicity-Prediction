import re
import pandas as pd
from datasets import load_dataset
import os
from utils.logger import get_logger

def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Collapse multiple whitespace or newlines into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(config: dict) -> pd.DataFrame:
    dataset = load_dataset(
        config["data"]["dataset_name"],
        keep_in_memory=False,
        cache_dir=config["data"]["cache_dir"],
    )
    # Train & Val Dataset
    df = dataset[config["data"]["train_split"]].to_pandas()
    df.dropna(inplace=True)
    df["comment_text"] = df["comment_text"].apply(clean_text)

    # Test Dataset
    test_df = dataset[config["data"]["test_split"]].to_pandas()
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    df = df.sample(frac=0.005, random_state=config['training']['seed'])
    test_df = test_df.sample(frac=0.05, random_state=config['training']['seed'])
    return df, test_df

def save_cleaned(df: pd.DataFrame, config: dict) -> str:
    log = get_logger('preprocessor', config['paths']['logs_dir'])
    path = os.path.join(config['data']['cache_dir'], 'cleaned.csv')
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    df.to_csv(path, index=False)
    log.info(f"saved in path: {path}, with length: {len(df)}")

    return path