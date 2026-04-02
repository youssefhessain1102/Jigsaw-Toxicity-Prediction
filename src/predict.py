import os
import json
import torch
import numpy as np
from transformers import DistilBertTokenizer

from model import DistilBERTClassifier
from preprocessor import clean_text
from utils.constants import LABELS
from utils.logger import get_logger


class PredictService:
    def __init__(self, config):
        self.config = config
        self.log = get_logger("predict", config["paths"]["logs_dir"])

        self.model = DistilBERTClassifier(
            config["model"]["name"],
            dropout=config["model"]["dropout"],
            num_labels=config["model"]["num_labels"],
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_model_path = os.path.join(
            config["paths"]["model_save_dir"], config["paths"]["best_model_name"]
        )
        self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()
        self.log.info(f"Model loaded from {best_model_path} | Running on: {self.device}")

        # ── Load per-label thresholds ──────────────────────────────────────
        thresholds_path = os.path.join(
            config["paths"]["model_save_dir"], config["paths"]["thresholds_name"]
        )
        with open(thresholds_path, "r") as f:
            thresholds_dict = json.load(f)

        # Keep as a numpy array aligned to LABELS order for fast vectorised apply
        self.thresholds = np.array([thresholds_dict[l] for l in LABELS], dtype=np.float32)
        self.log.info(f"Thresholds loaded: { {l: thresholds_dict[l] for l in LABELS} }")

    def predict(self, text: str) -> dict:
        text = clean_text(text)
        encoded = self.tokenizer(
            [text],
            max_length=self.config["model"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()   # (1, 6)

        self.log.debug(f"Probs: {dict(zip(LABELS, probs[0].tolist()))}")

        binary_outputs = (probs[0] >= self.thresholds).astype(int)
        return dict(zip(LABELS, binary_outputs.tolist()))