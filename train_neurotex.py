import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from neurotex_model import NeuroTeX
from tokenizer_and_dataset import LatexTokenizer, Im2LatexDataset

# CONFIG
# Paths and training hyperparameters
DATA_DIR = "./dataset"
LABEL_FILE = "master_labels.json"
TOKENIZER_FILE = "tokenizer.json"
MODEL_FILE = "neurotex.pth"
CHECKPOINT_DIR = "checkpoints"
HISTORY_FILE = "training_history.csv"

BATCH_SIZE = 4
MAX_LEN = 64
EPOCHS = 30
VAL_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collate function for batching
# Pads sequences and stacks images into a batch
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = [l[:MAX_LEN] for l in labels]  # Truncate long sequences
    padded = pad_sequence(labels, batch_first=True, padding_value=0)    # Pad to same length
    return images, padded


# Training pipeline
def train_model():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create checkpoint dir if needed

    # Load all training formulas from JSON
    label_path = os.path.join(DATA_DIR, LABEL_FILE)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"{label_path} not found. Run data generation first.")

    with open(label_path, "r", encoding="utf-8") as f:
        formulas = list(json.load(f).values())

    print(f"Total formulas in training set: {len(formulas)}")

    # Tokenizer creation or load from file
    if os.path.exists(TOKENIZER_FILE):
        tokenizer = LatexTokenizer.load(TOKENIZER_FILE)
        print(f"Loaded tokenizer from {TOKENIZER_FILE}")
    else:
        tokenizer = LatexTokenizer()
        tokenizer.build_vocab(formulas)
        tokenizer.save(TOKENIZER_FILE)
        print(f"Built and saved tokenizer to {TOKENIZER_FILE}")

    # Load dataset and split into training and validation
    full_dataset = Im2LatexDataset(DATA_DIR, tokenizer=tokenizer, labels_file=LABEL_FILE)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Total training samples seen so far: {len(full_dataset)}")

    # Model and optimizer setup
    model = NeuroTeX(vocab_size=tokenizer.vocab_size()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0    # Counts epochs without improvement

    # Resume from checkpoint if exists
    if os.path.exists(MODEL_FILE):
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed training from epoch {start_epoch}")

    # Loss function, ignoring PAD tokens
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        total_train_loss = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            inputs = targets[:, :-1].to(DEVICE)  # All except last token
            labels = targets[:, 1:].to(DEVICE)   # All except first token

            outputs = model(images, inputs) # Forward pass
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                inputs = targets[:, :-1].to(DEVICE)
                labels = targets[:, 1:].to(DEVICE)

                outputs = model(images, inputs)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Log training progress to CSV
        with open(HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow([epoch, avg_train_loss, avg_val_loss])

        # Save model if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint, MODEL_FILE)
            torch.save(checkpoint, f"{CHECKPOINT_DIR}/neurotex_epoch_{epoch:03d}.pth")
            print(f"Saved best model to {MODEL_FILE} and checkpoint to versioned file")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

# Entry point
if __name__ == "__main__":
    train_model()
