from model import DistilBERTClassifier
from preprocessor import clean_text
from transformers import DistilBertTokenizer
import torch
import os

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class PredictService:
    def __init__(self, config, threshold):
        self.config = config
        self.threshold = threshold

        self.model = DistilBERTClassifier(
            config["model"]["name"],
            dropout=config["model"]["dropout"],
            num_labels=config["model"]["num_labels"],
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path = os.path.join(
            config["paths"]["model_save_dir"], config["paths"]["best_model_name"]
        )

        self.model.load_state_dict(torch.load(self.best_model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        text = clean_text(text)
        text = self.tokenizer(
            [text],
            max_length=self.config["model"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

        binary_outputs = (probs >= self.threshold).astype(int)
        output = dict(zip(LABELS, binary_outputs[0]))

        return output
