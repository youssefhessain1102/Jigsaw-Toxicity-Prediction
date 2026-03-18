import os
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on {self.device}")

        self.pos_weights = torch.tensor(
            list(config["training"]["pos_weights"].values()), dtype=torch.float32
        )

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.total_steps = len(train_loader) * config["training"]["epochs"]
        self.warmup_steps = int(self.total_steps * config["training"]["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_training_steps=self.total_steps,
            num_warmup_steps=self.warmup_steps,
        )

        # self.scaler = GradScaler("cuda", enabled=config["training"]["fp16"])
        self.accum_steps = config["training"]["gradient_accumulation_steps"]

        self.model_save_dir = config["paths"]["model_save_dir"]
        self.best_model_name = config["paths"]["best_model_name"]
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.best_val_loss = float("inf")

    def train(self):
        for epoch in range(self.config["training"]["epochs"]):

            print("=" * 50)
            print(f"Epoch {epoch + 1} / {self.config['training']['epochs']}")
            print("=" * 50)

            train_loss = self._train_epoch()
            val_loss, auc_roc_score, _f1_score = self._validate()

            print(f"Train loss: {train_loss:.4f}")
            print(f"Validation: {val_loss:.4f}")
            print(f"auc_roc_score: {auc_roc_score:.4f}")
            print(f"F1 Score: {_f1_score:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model()

                print(f"model saved with Validation loss: {self.best_val_loss:.4f}")

    def _train_epoch(self):
        
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels) / self.accum_steps
            loss.backward()

            if (step + 1) % self.accum_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            all_preds_binary = (all_preds >= 0.5).astype(int)

            f1score = f1_score(
                all_labels, all_preds_binary, average="micro", zero_division=0
            )
            auc_roc_score = roc_auc_score(all_labels, all_preds_binary, average="macro")

            return total_loss / len(self.val_loader), auc_roc_score, f1score

    def _save_model(self):
        path = os.path.join(self.model_save_dir, self.best_model_name)
        torch.save(self.model.state_dict(), path)
