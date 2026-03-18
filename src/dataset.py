import torch
from torch.utils.data import Dataset
import pandas as pd

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, context_length):
        self.data = pd.read_csv(df)
        self.tokenizer = tokenizer

        self.encoded_text = tokenizer(
            self.data["comment_text"].tolist(),
            max_length=context_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(self.data[LABELS].values, dtype=torch.float32)

    def __getitem__(self, index):
        return {
            "input_ids": self.encoded_text['input_ids'][index],
            "attention_mask": self.encoded_text['attention_mask'][index],
            "labels": self.labels[index]
        }
    def __len__(self):
        return len(self.data)
