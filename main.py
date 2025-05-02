import os
import json
import shutil
import random
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# CONFIG
NUM_ITERATIONS = 10                                         # Total number of training + evaluation loops
DATASET_DIR = Path("dataset")                               # Base dataset directory
TOKENIZER_FILE = "tokenizer.json"                           # Tokenizer path
MODEL_FILE = "neurotex.pth"                                 # Model checkpoint path
MASTER_LABELS_FILE = DATASET_DIR / "master_labels.json"     # Cumulative label file
LABELS_FILE = DATASET_DIR / "labels.json"                   # Newly generated labels (to be merged)
IMAGES_DIR = DATASET_DIR / "images"                         # Directory of LaTeX images
HISTORY_CSV = "similarity_history.csv"                      # File to record iteration metrics

# Utilities
def run_command(name, cmd_list):
    """
    Executes a subprocess command and logs its stdout/stderr.
    """
    print(f"\n--- {name} ---")
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:\n" + result.stderr)
    return result

def merge_labels():
    """
    Appends new `labels.json` entries into `master_labels.json`, preserving existing data.
    """
    if not LABELS_FILE.exists():
        print("No labels.json to merge")
        return

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        new_labels = json.load(f)

    if MASTER_LABELS_FILE.exists():
        with open(MASTER_LABELS_FILE, "r", encoding="utf-8") as f:
            master = json.load(f)
    else:
        master = {}

    master.update(new_labels)
    with open(MASTER_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print(f"Merged {len(new_labels)} new labels into master_labels.json")

def predict_random_image():
    """
    Picks a random image from the dataset and runs a one-off prediction for it.
    """
    if not IMAGES_DIR.exists():
        print("No image directory found.")
        return
    files = list(IMAGES_DIR.glob("*.png"))
    if not files:
        print("No image files found.")
        return
    sample = random.choice(files)
    print(f"\n--- Predicting random image: {sample.name} ---")
    subprocess.run(["python", "predict_latex.py", str(sample)])

def save_checkpoint(iter_num):
    """
    Saves the current model and tokenizer with the iteration number appended.
    """
    iter_model = f"neurotex_iter_{iter_num:02d}.pth"
    iter_tokenizer = f"tokenizer_iter_{iter_num:02d}.json"
    if os.path.exists(MODEL_FILE):
        shutil.copy(MODEL_FILE, iter_model)
        print(f"Saved model: {iter_model}")
    if os.path.exists(TOKENIZER_FILE):
        shutil.copy(TOKENIZER_FILE, iter_tokenizer)
        print(f"Saved tokenizer: {iter_tokenizer}")

def parse_similarity(output):
    """
    Extracts the average normalized and raw similarity from output text.

    Returns:
        tuple: (normalized_similarity, raw_similarity)
    """
    norm_score = None
    raw_score = None
    for line in output.splitlines():
        if "Average Normalized Similarity" in line:
            try:
                norm_score = float(line.split(":")[-1].strip())
            except:
                pass
        elif "Average Raw Similarity" in line:
            try:
                raw_score = float(line.split(":")[-1].strip())
            except:
                pass
    return norm_score, raw_score

def append_to_history_csv(iter_num, norm_score, raw_score):
    """
    Appends one row to similarity_history.csv for long-term tracking.

    Args:
        iter_num (int): Current iteration number.
        norm_score (float): Normalized similarity score.
        raw_score (float): Raw similarity score.
    """
    header = ["iteration", "normalized_similarity", "raw_similarity"]
    exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([iter_num, f"{norm_score:.4f}", f"{raw_score:.4f}"])

# Main Iteration Loop
def main():
    """
    Orchestrates multiple iterations of data generation, training, evaluation,
    and logging to observe model improvement over time.
    """
    norm_scores = []
    raw_scores = []

    for i in range(1, NUM_ITERATIONS + 1):
        print(f"\n================ Iteration {i} of {NUM_ITERATIONS} ================\n")

        # Generate new data and merge it into master dataset
        run_command("Generating synthetic data", ["python", "create_data.py"])
        merge_labels()

        # Train model on full master dataset
        run_command("Training model", ["python", "train_neurotex.py"])

        # Run predictions and evaluate performance
        result = run_command("Running inference and evaluation",
                             ["python", "inference_neurotex.py", "--iteration", str(i)])
                             
        norm_score, raw_score = parse_similarity(result.stdout)

        # Store results (or fallback to 0.0 if missing)
        norm_scores.append(norm_score if norm_score is not None else 0.0)
        raw_scores.append(raw_score if raw_score is not None else 0.0)

        # Save to CSV log
        append_to_history_csv(i, norm_scores[-1], raw_scores[-1])

        # Sanity check prediction
        predict_random_image()

        # Save model/tokenizer snapshot for this iteration
        save_checkpoint(i)

    # Plot performance over time
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_ITERATIONS + 1), norm_scores, marker='o')
    plt.title("Average Normalized Similarity Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Similarity")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("progress_plot.png")
    print("Saved progress_plot.png")

    # Summary
    print("\nAll iterations complete!")
    print("Normalized Similarities:", [round(s, 3) for s in norm_scores])

if __name__ == "__main__":
    main()
