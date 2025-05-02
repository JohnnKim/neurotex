import os
import json
import csv
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from difflib import SequenceMatcher
from tokenizer_and_dataset import LatexTokenizer
from neurotex_model import NeuroTeX
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction
def predict_image(image_path, model, tokenizer, transform, max_len=64):
    """
    Predicts the LaTeX formula from an input image using the trained model.

    Args:
        image_path (str): Path to the input formula image.
        model (torch.nn.Module): The trained NeuroTeX model.
        tokenizer (LatexTokenizer): Tokenizer to decode predicted tokens.
        transform: Transformations to apply to the input image.
        max_len (int): Maximum number of tokens to generate.

    Returns:
        str: The predicted LaTeX string.
    """
    image = Image.open(image_path).convert("RGB")
    # Add batch dimension and send to device
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Start with beginning-of-sequence token
    tokens = [tokenizer.vocab["<BOS>"]]
    # Predict tokens one by one
    for _ in range(max_len):
        # Current partial sequence
        input_seq = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Get predictions from model
            output = model(image, input_seq)
            # Choose the most likely next token
        next_token = output[0, -1].argmax(-1).item()
        if next_token == tokenizer.vocab["<EOS>"]:
            break   # Stop if end-of-sequence token is reached
        tokens.append(next_token)

    return tokenizer.decode(tokens) # Convert token IDs back to LaTeX string

def similarity(a, b):
    """
    Computes similarity between two strings using fuzzy matching.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    return SequenceMatcher(None, a, b).ratio()

def normalize_latex(expr):
    """
    Removes irrelevant LaTeX formatting tokens to normalize expression comparison.

    Args:
        expr (str): Input LaTeX string.

    Returns:
        str: Normalized LaTeX string without spacing and cosmetic tokens.
    """
    expr = expr.replace(" ", "").replace("\\,", "").replace("\\ ", "")
    expr = expr.replace(r"\quad", "").replace(r"\!", "")
    return expr

def plot_similarity(scores, out_file):
    """
    Plots similarity scores for each sample and saves as an image.

    Args:
        scores (list of float): List of similarity scores.
        out_file (str): Output filename for the plot image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(scores, marker='o')
    plt.title("Normalized Prediction Similarity per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Similarity Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(out_file)
    print(f"Saved similarity plot: {out_file}")

def main():
    """
    Main entry point:
    - Loads model and tokenizer.
    - Predicts LaTeX for each image in dataset.
    - Compares predictions to ground truth.
    - Saves results to CSV and generates similarity plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=None, help="Optional iteration suffix for outputs")
    args = parser.parse_args()
    suffix = f"_iter_{args.iteration:02d}" if args.iteration is not None else ""

    # Load tokenizer from disk
    tokenizer = LatexTokenizer.load("tokenizer.json")
    print("Loaded tokenizer")

    # Load image label mappings
    with open("dataset/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Load trained model weights
    model = NeuroTeX(vocab_size=tokenizer.vocab_size()).to(DEVICE)
    checkpoint = torch.load("neurotex.pth", map_location=DEVICE)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Loaded model")
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    similarities = []         # Normalized similarity scores
    raw_similarities = []     # Raw similarity scores
    results = []              # Output rows for CSV

    print("\n--- Running Inference ---\n")

    # Loop through each image and run prediction
    for i, (filename, gt) in enumerate(labels.items()):
        img_path = os.path.join("dataset/images", filename)
        try:
            pred = predict_image(img_path, model, tokenizer, transform)
            raw_score = similarity(pred, gt)
            norm_score = similarity(normalize_latex(pred), normalize_latex(gt))
        except Exception as e:
            pred = "[ERROR]"
            raw_score = norm_score = 0.0
            print(f"Error processing {filename}: {e}")

        raw_similarities.append(raw_score)
        similarities.append(norm_score)
        results.append({
            "filename": filename,
            "prediction": pred,
            "ground_truth": gt,
            "raw_similarity": f"{raw_score:.2f}",
            "normalized_similarity": f"{norm_score:.2f}"
        })

        print(f"[{i+1:03d}] {filename}")
        print(f"   Predicted   : {pred}")
        print(f"   Ground Truth: {gt}")
        print(f"   Raw Sim.    : {raw_score:.2f}")
        print(f"   Norm. Sim.  : {norm_score:.2f}\n")

    # Calculate averages
    avg_raw = sum(raw_similarities) / len(raw_similarities)
    avg_norm = sum(similarities) / len(similarities)
    print(f"Average Raw Similarity   : {avg_raw:.3f}")
    print(f"Average Normalized Similarity: {avg_norm:.3f}")

    # Save all results to a CSV file
    csv_file = f"inference_results{suffix}.csv"
    with open(csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "ground_truth", "raw_similarity", "normalized_similarity"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {csv_file}")

    # Plot similarity scores
    plot_file = f"similarity_plot{suffix}.png"
    plot_similarity(similarities, plot_file)

if __name__ == "__main__":
    main()
