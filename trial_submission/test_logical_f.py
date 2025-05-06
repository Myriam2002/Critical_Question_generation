#!/usr/bin/env python3
"""
test_fallacy_classification.py

A script to test the q3fer/distilbert-base-fallacy-classification model from Hugging Face.
"""

import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Test fallacy classification model")
    parser.add_argument(
        '--input_file',
        type=str,
        help="Path to a text file containing interventions (one per line)"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='q3fer/distilbert-base-fallacy-classification',
        help="Hugging Face model name"
    )
    args = parser.parse_args()

    # Initialize the classification pipeline
    classifier = pipeline(
        'text-classification',
        model=args.model_name,
        return_all_scores=True
    )

    # Load texts to classify
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # TODO: Replace this placeholder with your intervention text
        texts = ["MT: \"Claire\u2019s absolutely right about that\nBut then the problem is that that form of capitalism wasn\u2019t generating sufficient surpluses\nAnd so therefore where did the money flow\nIt didn\u2019t flow into those industrial activities\n because in the developed world that wasn\u2019t making enough money\""]

    # Classify each text and display scores for all labels
    for text in texts:
        print(f"Input text: {text}\n")
        scores = classifier(text)[0]
        for entry in scores:
            label = entry['label']
            score = entry['score']
            print(f"  {label}: {score:.4f}")
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
