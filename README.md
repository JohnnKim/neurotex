# NeuroTeX

NeuroTeX is a Python based pipeline for converting images of mathematical formulas into LaTeX code using deep learning. It supports:

- Fully synthetic dataset generation using `matplotlib`  
- Iterative training with a ResNet-Transformer model  
- Versioned checkpoints and tokenizers  
- Accuracy evaluation across multiple runs  
- CLI-based image prediction and batch evaluation  

This repository is built for extensibility and modular experimentation.

---

## How It Works

NeuroTeX solves the image-to-LaTeX problem using a two-stage model:

- **Visual Encoder**: A pretrained ResNet-18 CNN extracts visual features from formula images.  
- **Sequence Decoder**: A Transformer autoregressively decodes LaTeX tokens using embeddings and attention.

---

## Directory Structure

```
├── checkpoints/               # Auto-saved model checkpoints (optional in Git)
├── dataset/
│   ├── images/                # Auto-generated formula images
│   ├── labels.json            # Current batch of image-label pairs
│   └── master_labels.json     # Cumulative training dataset
├── cleanup.py                 # Deletes all older iteration results
├── create_data.py             # Formula template generator (matplotlib-based)
├── inference_neurotex.py      # Evaluation script (batch similarity scoring)
├── main.py                    # Runs full N-iteration training/eval loop
├── neurotex_model.py          # ResNet encoder + Transformer decoder
├── predict_latex.py           # CLI: image → LaTeX prediction
├── reset_all.py               # Resets all models, data, plots, and results
├── train_neurotex.py          # Core model training script (with early stopping)
├── tokenizer_and_dataset.py   # Tokenizer + PyTorch Dataset wrapper
├── *.pth, *.json, *.csv       # Model weights, vocabularies, metrics (auto-generated)
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- Pillow

Install everything with:

```bash
pip install -r requirements.txt
```

---

## How to Use

### 1. Generate and Train (Full Pipeline)

```bash
python main.py
```

This runs multiple iterations:

- `create_data.py`: Generates random math expressions  
- `train_neurotex.py`: Trains on all `master_labels.json`  
- `inference_neurotex.py`: Evaluates and saves metrics  
- Saves `neurotex.pth` + `tokenizer.json`

---

### 2. Predict a Single Formula Image

```bash
python predict_latex.py dataset/images/eq_xxxxxxxx.png
```

Outputs the predicted LaTeX string to the console.

---

### 3. Reset All Data

```bash
python reset_all.py
```

Removes all generated models, images, plots, and CSVs.

---

## Evaluation Metrics

After each iteration, the system logs:

- **Average Raw Similarity**: character-level string match  
- **Average Normalized Similarity**: ignores whitespace/spacing  

These are saved in:

- `similarity_history.csv`
- `similarity_plot_iter_XX.png`
- `progress_plot.png`

---

## Model Architecture

**Encoder (ResNet18):**

- Pretrained on ImageNet  
- Removes final FC layer  
- Outputs spatial feature map

**Decoder (Transformer):**

- Embeds tokens and adds position encoding  
- Applies 4-layer decoder stack with 8 heads  
- Projects to vocabulary size using Linear

---

## Tokenizer

**Features:**

- Regex-based LaTeX tokenization  
- Special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`  
- Bidirectional mapping for encoding/decoding  
- Tokenizers are versioned per iteration (e.g., `tokenizer_iter_10.json`)

---

## Training Details

- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 4  
- Early stopping: 5 epochs patience  
- Max token length: 64  
- Training runs on GPU if available

---

## GitHub Suggestions

Add a `.gitignore` with:

```
__pycache__/
*.pth
*.csv
*.png
dataset/images/
dataset/*.json
tokenizer*.json
```

---

## Credits

Created by **John Kim**  
Model design, iteration system, and full pipeline implementation

---

## Contributing

Pull requests welcome. For major changes, open an issue first.

---

## License

MIT License
