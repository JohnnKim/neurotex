import os
import json
import random
import uuid
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# Configurable Constants
# Number of formulas to generate per run
DEFAULT_NUM_FORMULAS = 50
OUTPUT_DIR = Path("dataset/images")
MASTER_LABELS_FILE = Path("dataset/master_labels.json")
LABELS_FILE = Path("dataset/labels.json")

# Configure matplotlib to render math using mathtext
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 18

# Template Pools
# List of possible variables to use in formulas
VARIABLES = ["x", "y", "z", "a", "b", "n"]

# LaTeX templates with placeholders `{v}` for variable and `{n}` for number
TEMPLATES = [
    # Simple
    r"{v} + {n}",
    r"{v} - {n}",
    r"{n} - {v}",
    r"{n}{v}",
    r"{v} = {n}",
    r"{v}^2",
    r"{v}^{n}",
    r"{v} + {v} + {n}",

    # Fractions and roots
    r"\frac{{{v} + {n}}}{{{n}}}",
    r"\frac{{{v}}}{{{n}}}",
    r"\frac{{d{v}}}{{dt}} = {n}{v}",
    r"\sqrt{{{v}^2 + {n}}}",
    r"\sqrt{{{n}}}",

    # Calculus
    r"\int_0^{{{n}}} {v}^2 \, d{v}",
    r"\frac{{dy}}{{dx}} = {v}^2 + {v}",
    r"\int {v} \, d{v} = \frac{{{v}^2}}{{2}} + C",

    # Limits
    r"\lim_{{{v} \to 0}} \frac{{\sin {v}}}{{{v}}}",
    r"\lim_{{{v} \to \infty}} \frac{{1}}{{{v}}}",
    r"\lim_{{{v} \to {n}}} {v}^2 + {v}",

    # Logs / Exponentials
    r"\log_{{2}}({v}) = {n}",
    r"\ln({v})",
    r"e^{{{v}}} = \sum_{{n=0}}^{{\infty}} \frac{{{v}^n}}{{n!}}",

    # Series / Summations
    r"\sum_{{i=1}}^{{n}} i = \frac{{n(n+1)}}{{2}}",
    r"\sum_{{k=0}}^{{\infty}} \frac{{x^k}}{{k!}}",
    r"\sum_{{{v}=1}}^{{\infty}} \frac{{1}}{{{v}^2}}",

    # Quadratics / Polynomials
    r"{v}^2 + {v} + 1",
    r"{v}^2 - {v} + 1 = 0",
    r"{v}^3 - 3{v} + 2 = 0",
    r"{v} = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}",

    # Trig / Physics
    r"{v}(t) = A \sin(\omega t + \phi)",
    r"F = ma",
    r"E = mc^2",
    r"\theta = \frac{{s}}{{r}}",
    r"\sin^2 \theta + \cos^2 \theta = 1",

    # Subscripts / Sequences
    r"{v}_{{1}} = 5, \quad {v}_{{2}} = -1",
    r"{v}_n = {v}_{{n-1}} + {v}_{{n-2}}",
    r"a_n = \frac{{1}}{{n^2}}",

    # Misc Complex Forms
    r"\left( \frac{{{v}}}{{{n}}} \right)^2",
    r"{v} = A e^{{-kt}}",
    r"y = mx + b",
    r"{v} = r \cos(\theta)",
    r"\vec{{F}} = q \vec{{E}} + q \vec{{v}} \times \vec{{B}}"
]

# Formula Generator
def generate_formulas(num=DEFAULT_NUM_FORMULAS, seed=None):
    """
    Randomly generate a dictionary of LaTeX formulas using the predefined templates.
    Each entry maps a filename to a formula.
    """
    if seed is not None:
        random.seed(seed)

    formulas = {}
    while len(formulas) < num:
        v = random.choice(VARIABLES)
        n = random.randint(1, 9)
        template = random.choice(TEMPLATES)
        latex = template.format(v=v, n=n)
        filename = f"eq_{uuid.uuid4().hex[:8]}.png"
        formulas[filename] = latex
    return formulas

# Renderer
def render_and_save(formulas, output_dir):
    """
    Render LaTeX formulas into images and save them to the specified output directory.
    Skips malformed formulas and logs any rendering errors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    good_formulas = {}
    for filename, formula in formulas.items():
        try:
            fig, ax = plt.subplots(figsize=(4, 1.5))
            ax.text(0.5, 0.5, f"${formula}$", ha='center', va='center')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=200)
            plt.close()
            good_formulas[filename] = formula
        except Exception as e:
            print(f"Skipped malformed LaTeX: {formula}\n   {e}")
    return good_formulas

# Label Saving
def update_labels(new_labels):
    """
    Append new label mappings to master_labels.json and overwrite labels.json
    with only the latest batch (for short-term evaluation).
    """
    if MASTER_LABELS_FILE.exists():
        with open(MASTER_LABELS_FILE, "r", encoding="utf-8") as f:
            master = json.load(f)
    else:
        master = {}

    master.update(new_labels)

    with open(MASTER_LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_labels, f, indent=2)

# Main
def main():
    """
    Entry point for synthetic data generation.
    Parses command-line arguments, generates formulas, renders images,
    and updates label files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_FORMULAS, help="Number of formulas to generate")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating {args.num} formulas (seed={args.seed})...")
    raw_formulas = generate_formulas(num=args.num, seed=args.seed)
    good_formulas = render_and_save(raw_formulas, OUTPUT_DIR)

    if not good_formulas:
        print("No valid formulas rendered. Aborting.")
        return

    update_labels(good_formulas)
    print(f"Saved {len(good_formulas)} formulas to {OUTPUT_DIR} and updated master_labels.json")

if __name__ == "__main__":
    main()
