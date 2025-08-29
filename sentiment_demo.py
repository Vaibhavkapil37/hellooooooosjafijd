#!/usr/bin/env python3
# sentiment_demo.py
from transformers import pipeline
import json

def main():
    print("Soulance — Sentiment Analyzer (Hugging Face pipeline)")
    clf = pipeline("sentiment-analysis")  # auto-downloads a small fine-tuned model
    while True:
        try:
            text = input("\nEnter text (or type 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not text:
            print("Type a sentence to analyze.")
            continue
        if text.lower() in ("exit","quit"):
            print("Bye.")
            break

        scores = clf(text, return_all_scores=True)[0]
        probs = {p['label']: round(p['score'],4) for p in scores}
        best = max(scores, key=lambda x: x['score'])
        print("\n--- Result ---")
        print(f"Input : {text}")
        print(f"Sentiment: {best['label']}  (confidence: {best['score']:.4f})")
        print("Full distribution:", json.dumps(probs))
        print("----------------\n")

if __name__ == "__main__":
    main()
