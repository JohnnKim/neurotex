import os
import json
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# LatexTokenizer
# A class to tokenize, encode, decode LaTeX formulas and manage a vocabulary.
class LatexTokenizer:
    def __init__(self, tokens=None, special_tokens=None):
        # Special tokens: padding, start, end, unknown
        self.special_tokens = special_tokens or ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.tokens = tokens or []
        self._rebuild_vocab()

    def _rebuild_vocab(self):
        # Build forward and inverse vocab mappings
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens + self.tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def tokenize(self, formula):
        # Tokenizes a formula into LaTeX-relevant pieces using regex
        return re.findall(r"[a-zA-Z]+|\\[a-zA-Z]+|[0-9]+|[\^\_\=\+\-\*/\{\}\(\)\[\],]|\\\\|&|%|\$|<PAD>|<BOS>|<EOS>|<UNK>", formula)

    def build_vocab(self, formulas):
        # Builds vocabulary from a list of formulas
        token_set = set()
        for formula in formulas:
            token_set.update(self.tokenize(formula))
        self.tokens = sorted(token_set)
        self._rebuild_vocab()

    def encode(self, formula):
        # Converts formula to list of token IDs, wrapped in BOS/EOS
        tokens = self.tokenize(formula)
        return [self.vocab["<BOS>"]] + [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens] + [self.vocab["<EOS>"]]

    def decode(self, ids):
        # Converts list of token IDs back into a formula string
        tokens = [self.inv_vocab.get(i, "<UNK>") for i in ids if i not in [self.vocab["<PAD>"], self.vocab["<BOS>"], self.vocab["<EOS>"]]]
        return ' '.join(tokens).replace(" \\", "\\").strip()

    def vocab_size(self):
        # Returns the number of tokens in the vocabulary
        return len(self.vocab)

    def save(self, path):
        # Saves tokenizer config to a JSON file
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokens": self.tokens, "special_tokens": self.special_tokens}, f)

    @staticmethod
    def load(path):
        # Loads tokenizer config from a JSON file
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LatexTokenizer(tokens=data["tokens"], special_tokens=data["special_tokens"])

# Im2LatexDataset
# A PyTorch Dataset class that loads formula images and their LaTeX labels.
class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None, labels_file="labels.json"):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for Im2LatexDataset")

        self.data_dir = data_dir
        self.tokenizer = tokenizer

        # Default image transform if not provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Load labels from JSON file
        labels_path = os.path.join(data_dir, labels_file)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"{labels_path} not found.")

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.images = list(self.annotations.keys())

    def __len__(self):
        # Total number of image-label pairs
        return len(self.images)

    def __getitem__(self, idx):
        # Loads image and encoded LaTeX label
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, "images", img_name)

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Get formula and encode it to token IDs
        formula = self.annotations[img_name]
        label = self.tokenizer.encode(formula)

        return image, torch.tensor(label)
