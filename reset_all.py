import os
import shutil

# List of files to delete for a full reset (models, tokenizers, results, labels)
delete_files = [
    "neurotex.pth",                      # Main trained model
    "inference_results.csv",             # Last inference output
    "similarity_plot.png",               # Last similarity graph
    "tokenizer.json",                    # Current tokenizer
    *[f"neurotex_iter_{i:02d}.pth" for i in range(1, 6)],       # Checkpointed models
    *[f"tokenizer_iter_{i:02d}.json" for i in range(1, 6)],     # Checkpointed tokenizers
    "dataset/labels.json",              # Current session's labels
    "dataset/master_labels.json"        # Accumulated master labels
]

# Delete each listed file if it exists
for file in delete_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted {file}")

# Clear all generated formula images
image_dir = "dataset/images"
if os.path.exists(image_dir):
    for img in os.listdir(image_dir):
        if img.endswith(".png"):
            os.remove(os.path.join(image_dir, img))
    print("Cleared dataset/images")

print("\nReset complete.")
