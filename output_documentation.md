## 📁 NeuroTeX Data Files Documentation

sampleData contains data that was generated through multiple iterations on my device.
This document explains the purpose and structure of these three core files used in the **NeuroTeX** image-to-LaTeX pipeline:

---

### 📘 `master_labels.json`

**Type:** JSON dictionary

```json
{
  "eq_83f3b9ea.png": "z^3 - 3z + 2 = 0",
  "eq_0cf96a0b.png": "\\int_0^{7} b^2 \\, db"
}
```

**Purpose:** Stores all known image–LaTeX pairs used for training.

**Details:**

* Each key is the filename of a rendered formula image (e.g., `eq_XXXX.png`).
* Each value is the corresponding LaTeX formula string.
* It is updated incrementally during each synthetic data generation step.

**Used by:**

* `train_neurotex.py`
* `Im2LatexDataset` (via `tokenizer_and_dataset.py`)

---

### 📘 `tokenizer.json`

**Type:** JSON object with token mappings

```json
{
  "tokens": ["+", "-", "^", "\\int", "x", "1", ...],
  "special_tokens": ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
}
```

**Purpose:** Stores the vocabulary used for tokenizing LaTeX strings.

**Details:**

* `tokens`: All unique LaTeX symbols seen during training.
* `special_tokens`: Reserved keywords to manage padding, sequence boundaries, and unknowns.
* Used for converting formulas to and from token sequences.
* Versioned after each iteration as `tokenizer_iter_XX.json`.

**Used by:**

* `train_neurotex.py`
* `predict_latex.py`
* `inference_neurotex.py`

---

### 📘 `neurotex.pth`

**Type:** PyTorch checkpoint (binary file)

**Structure:**

```python
{
  "epoch": 39,
  "model_state": model.state_dict(),
  "optimizer_state": optimizer.state_dict(),
  "best_val_loss": 0.061
}
```

**Purpose:** Stores the model weights, optimizer state, and training metadata.

**Details:**

* `model_state`: Trained weights of the ResNet-Transformer model.
* `optimizer_state`: (Optional) allows resuming training.
* `epoch`: Last completed epoch.
* `best_val_loss`: Tracked validation loss for early stopping.

**Used by:**

* `train_neurotex.py`
* `predict_latex.py`
* `main.py`

---