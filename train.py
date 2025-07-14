# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import json
import numpy as np
from pathlib import Path

# === Load Config ===
with open("train_config.json") as f:
    config = json.load(f)

embedding_dim = config.get("embedding_dim", 128)
hidden_dim = config.get("hidden_dim", 256)
batch_size = config.get("batch_size", 32)
epochs = config.get("epochs", 10)
learning_rate = config.get("learning_rate", 0.001)
max_seq_len = config.get("max_seq_len", 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenized Data ===
with open("data/tokenized_prompts.pkl", "rb") as f:
    prompts = pickle.load(f)

with open("data/tokenized_responses.pkl", "rb") as f:
    responses = pickle.load(f)

# === Load Vocabulary ===
with open("data/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f.readlines()]
vocab_size = len(vocab)
pad_idx = vocab.index("<PAD>") if "<PAD>" in vocab else 0

# === Pad Sequences ===
def pad_sequence(seq, max_len, pad_value):
    return seq[:max_len] + [pad_value] * max(0, max_len - len(seq))

prompts_padded = [pad_sequence(p, max_seq_len, pad_idx) for p in prompts]
responses_padded = [pad_sequence(r, max_seq_len, pad_idx) for r in responses]

# === PyTorch Dataset ===
class ChatDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = ChatDataset(prompts_padded, responses_padded)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Model Definition ===
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)  # [batch, seq_len, vocab_size]

model = Seq2SeqModel(vocab_size, embedding_dim, hidden_dim, pad_idx).to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
print("🧠 Training started...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)  # [B, T, V]
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy calculation (ignore PAD)
        preds = outputs.argmax(dim=-1)
        mask = batch_targets != pad_idx
        correct = (preds == batch_targets) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    print(f"[📈] Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} — Accuracy: {accuracy:.4f}")

# === Save Model ===
Path("model").mkdir(exist_ok=True)
torch.save(model.state_dict(), "model/model_weights.pth")
print("\n✅ Training complete! Model saved to model/model_weights.pth")
