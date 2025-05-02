NeuroTeX

NeuroTeX is a Python-based pipeline for converting images of mathematical formulas into LaTeX code using deep learning. It supports fully synthetic dataset generation, iterative training using a ResNet-Transformer architecture, and accuracy evaluation across multiple runs.

This repository is structured for extensibility, performance tracking, and modular experimentation.

NeuroTeX solves the image to LaTeX problem using a two-part architecture:
Visual Encoder: A ResNet-18 CNN processes input images into feature maps.
Sequence Decoder: A Transformer decoder predicts the LaTeX sequence one token at a time using learned embeddings and attention.

This project supports:
Synthetic data generation using matplotlib
Iterative training and evaluation with early stopping
Similarity scoring (raw + normalized)
Versioned checkpoints and tokenizers
CLI-based image inference

Directory Structure
.
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

Requirements:
Python 3.8+
PyTorch
torchvision
matplotlib
Pillow

Install everything with:
pip install -r requirements.txt

How to Use
1. Generate and Train (Full Pipeline)
python main.py
This runs multiple iterations:
create_data.py: Generates random math expressions
train_neurotex.py: Trains on all master_labels.json
inference_neurotex.py: Evaluates and saves metrics
Saves neurotex.pth + tokenizer.json

2. Predict a Single Formula Image
python predict_latex.py dataset/images/eq_xxxxxxxx.png
Outputs the predicted LaTeX string to the console.

3. Reset All Data
python reset_all.py
Removes all generated models, images, plots, and CSVs.

Evaluation Metrics
After each iteration, the system logs:
Average Raw Similarity: character-level string match
Average Normalized Similarity: ignores whitespace/spacing

These are saved in:
similarity_history.csv
similarity_plot_iter_XX.png
progress_plot.png

Model Architecture
Encoder (ResNet18):
Pretrained on ImageNet
Removes final FC layer
Outputs spatial feature map
Decoder (Transformer):
Embeds tokens and adds position encoding
Applies 4-layer decoder stack with 8 heads
Projects to vocabulary size using Linear

Tokenizer
Features:
Regex-based LaTeX tokenization
Special tokens: <PAD>, <BOS>, <EOS>, <UNK>
Bidirectional mapping for encoding/decoding
Tokenizers are versioned per iteration (e.g., tokenizer_iter_10.json).

Training Details
Optimizer: Adam
Learning rate: 1e-4
Batch size: 4
Early stopping: 5 epochs patience
Max token length: 64
Training runs on GPU if available.

GitHub Suggestions
Add .gitignore

__pycache__/
*.pth
*.csv
*.png
dataset/images/
dataset/*.json
tokenizer*.json

Commit
git init
git add .
git commit -m "Initial commit for NeuroTeX"
git remote add origin https://github.com/YOUR_USERNAME/neurotex.git
git push -u origin main

Credits
Created by John Kim
Model design, iteration system, and full pipeline implementation

Contributing
Pull requests welcome. For major changes, open an issue first.

License
MIT License