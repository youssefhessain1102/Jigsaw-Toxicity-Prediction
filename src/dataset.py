import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.constants import LABELS

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.data = pd.read_csv(df)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = torch.tensor(self.data[LABELS].values, dtype=torch.float32)

    def __getitem__(self, index):
        text = self.data.iloc[index]['comment_text']
        encoded_text = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded_text['input_ids'].squeeze(0),
            "attention_mask": encoded_text['attention_mask'].squeeze(0),
            "labels": self.labels[index]
        }
    def __len__(self):
        return len(self.data)
