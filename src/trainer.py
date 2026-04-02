import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
import numpy as np
from utils.logger import get_logger

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.log = get_logger('trainer', config['paths']['logs_dir'])
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.log.info(f"Training on: {self.device}")

        # fp16 only makes sense on CUDA — silently disabled on CPU
        self.use_fp16 = config["training"]["fp16"] and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_fp16)
        self.log.info(f"Mixed precision (fp16): {'enabled' if self.use_fp16 else 'disabled'}")

        # pos_weights must be on the same device as logits
        self.pos_weights = torch.tensor(
            list(config["training"]["pos_weights"].values()), dtype=torch.float32
        ).to(self.device)

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

        self.accum_steps = config["training"]["gradient_accumulation_steps"]

        self.model_save_dir = config["paths"]["model_save_dir"]
        self.best_model_name = config["paths"]["best_model_name"]
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.best_val_loss = float("inf")

    def train(self):
        for epoch in range(self.config["training"]["epochs"]):
            self.log.info("=" * 50)
            self.log.info(f"Epoch {epoch + 1} / {self.config['training']['epochs']}")
            self.log.info("=" * 50)

            train_loss = self._train_epoch()
            val_loss, auc, f1 = self._validate()

            self.log.info(f"Train Loss : {train_loss:.4f}")
            self.log.info(f"Val Loss   : {val_loss:.4f}")
            self.log.info(f"AUC-ROC    : {auc:.4f}")
            self.log.info(f"F1 (macro) : {f1:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model()
                self.log.info(f"✓ Model saved — best val loss: {self.best_val_loss:.4f}")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            # autocast: runs in float16 on CUDA, float32 on CPU (no-op)
            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss   = self.criterion(logits, labels) / self.accum_steps

            # scaler.scale() is a no-op when fp16 is disabled
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

        # flush any remaining accumulated gradients (last incomplete batch)
        remainder = len(self.train_loader) % self.accum_steps
        if remainder != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        all_probs  = []
        all_labels = []

        with torch.inference_mode():
            for batch in self.val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_fp16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss   = self.criterion(logits, labels)

                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().float().numpy()  # cast back to fp32 for numpy
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_probs  = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        binary_preds = (all_probs >= 0.5).astype(int)

        f1  = f1_score(all_labels, binary_preds, average="macro", zero_division=0)
        auc = roc_auc_score(all_labels, all_probs, average="macro")  # raw probs, not binary

        return total_loss / len(self.val_loader), auc, f1

    def _save_model(self):
        path = os.path.join(self.model_save_dir, self.best_model_name)
        torch.save(self.model.state_dict(), path)
