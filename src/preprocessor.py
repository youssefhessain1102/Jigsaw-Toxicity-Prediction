import re
import pandas as pd
from datasets import load_dataset
import os

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(config: dict) -> pd.DataFrame:
    dataset = load_dataset(
        config["data"]["dataset_name"],
        keep_in_memory=False,
        cache_dir=config["data"]["cache_dir"],
    )
    df = dataset[config["data"]["train_split"]].select_columns(
        ["comment_text"] + LABELS
    ).to_pandas()
    
    df.dropna(inplace=True)
    df["comment_text"] = df["comment_text"].apply(clean_text)
    df = df[df["comment_text"].str.len() > 0].reset_index(drop=True)
    df = df.sample(frac=0.15, random_state=config['training']['seed'])

    return df

def save_cleaned(df: pd.DataFrame, config: dict) -> str:
    path = os.path.join(config['data']['cache_dir'], 'cleaned.csv')
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    df.to_csv(path)
    print(f"saved in path: {path}, with length: {len(df)}")

    return path