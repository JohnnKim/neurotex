import os
import argparse
import torch
from PIL import Image
from neurotex_model import NeuroTeX
from tokenizer_and_dataset import LatexTokenizer
import torchvision.transforms as transforms

# Set device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction Logic
def predict_image(image_path, model, tokenizer, transform, max_len=64):
    """
    Performs inference on a single image to generate a LaTeX sequence.

    Args:
        image_path (str): Path to the input image.
        model (NeuroTeX): The trained vision-to-LaTeX model.
        tokenizer (LatexTokenizer): Tokenizer for decoding predictions.
        transform (callable): Preprocessing transform for the image.
        max_len (int): Maximum number of tokens to decode.

    Returns:
        str: The predicted LaTeX expression as a string.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)    # Add batch dim
    # Initialize token sequence with <BOS> token
    tokens = [tokenizer.vocab["<BOS>"]]

    # Autoregressively generate tokens up to max_len or until <EOS>
    for _ in range(max_len):
        input_seq = torch.tensor(tokens).unsqueeze(0).to(DEVICE)    # [1, T]
        with torch.no_grad():
            output = model(image, input_seq)    # [1, T, vocab_size]
        next_token = output[0, -1].argmax(-1).item()    # Get most probable next token
        if next_token == tokenizer.vocab["<EOS>"]:
            break
        tokens.append(next_token)
    # Convert tokens to string
    return tokenizer.decode(tokens)

# CLI Entry Point
def main():
    """
    CLI interface for predicting LaTeX from a single image.
    Loads the tokenizer and model, runs inference, and prints output.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the formula image")
    parser.add_argument("--max_len", type=int, default=64, help="Max token length to decode")
    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}")
        return

    # Validate required files
    if not os.path.exists("tokenizer.json") or not os.path.exists("neurotex.pth"):
        print("Required files missing: tokenizer.json or neurotex.pth")
        return

    # Load checkpoint (supports both plain or dict-wrapped format)
    print("Loading model and tokenizer...")
    tokenizer = LatexTokenizer.load("tokenizer.json")
    model = NeuroTeX(vocab_size=tokenizer.vocab_size()).to(DEVICE)

    checkpoint = torch.load("neurotex.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()    # Set model to evaluation mode

    # Define image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    print(f"Predicting LaTeX for: {args.image_path}")
    try:
        prediction = predict_image(args.image_path, model, tokenizer, transform, max_len=args.max_len)
        print("\nPrediction:\n" + prediction)
    except Exception as e:
        print(f"Prediction failed: {e}")

# Run the CLI
if __name__ == "__main__":
    main()
