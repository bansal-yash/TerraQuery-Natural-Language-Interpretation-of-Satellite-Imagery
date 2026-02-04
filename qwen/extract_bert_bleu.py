#!/usr/bin/env python3
"""
Extract BERT-BLEU scores from saved bbscore_all.py JSON output.
Usage: python extract_bert_bleu.py <json_file>
"""

import argparse
import json


def extract_bert_bleu(json_file):
    """Extract and display BERT-BLEU scores from saved JSON file."""
    print(f"📄 Loading results from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # The last entry contains the metrics
    metrics_entry = data[-1]
    
    if 'metrics_finetuned' not in metrics_entry or 'metrics_base' not in metrics_entry:
        print("❌ Error: JSON file doesn't contain metrics in expected format")
        return
    
    metrics_ft = metrics_entry['metrics_finetuned']
    metrics_base = metrics_entry['metrics_base']
    
    print(f"\n{'='*80}")
    print("BERT-BLEU Scores")
    print(f"{'='*80}\n")
    
    print("🔹 Fine-tuned Model:")
    print(f"   BERT-BLEU: {metrics_ft['bert-bleu']:.6f}\n")
    
    print("🔸 Base Model:")
    print(f"   BERT-BLEU: {metrics_base['bert-bleu']:.6f}\n")
    
    # Compare
    diff_bleu = metrics_ft['bert-bleu'] - metrics_base['bert-bleu']
    if diff_bleu > 0:
        print(f"📊 Fine-tuned model is better by {diff_bleu:.6f} BERT-BLEU points")
    elif diff_bleu < 0:
        print(f"📊 Base model is better by {abs(diff_bleu):.6f} BERT-BLEU points")
    else:
        print(f"📊 Both models have equal BERT-BLEU scores")
    
    # Also show BERTScore if available
    if metrics_ft.get('bert-f1') is not None:
        print(f"\n{'='*80}")
        print("BERTScore (F1)")
        print(f"{'='*80}\n")
        print(f"🔹 Fine-tuned Model: {metrics_ft['bert-f1']:.4f}")
        print(f"🔸 Base Model:       {metrics_base['bert-f1']:.4f}")
    
    print(f"\n{'='*80}\n")
    
    # Print summary stats
    num_samples = len(data) - 1  # Exclude the metrics entry
    print(f"ℹ️  Total samples evaluated: {num_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract BERT-BLEU scores from bbscore_all.py JSON output'
    )
    parser.add_argument(
        'json_file',
        help='Path to the JSON file containing results'
    )
    
    args = parser.parse_args()
    extract_bert_bleu(args.json_file)


if __name__ == '__main__':
    main()
