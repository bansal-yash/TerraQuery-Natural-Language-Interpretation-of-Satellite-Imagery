import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load BERT-base model for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


def get_embedding(text):
    """Return mean pooled BERT embedding of a token sequence."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Properly mask padding tokens for mean pooling
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
    
    # Sum embeddings where mask is 1, divide by number of real tokens
    masked_embeddings = embeddings * attention_mask
    sum_embeddings = masked_embeddings.sum(dim=1)  # (1, hidden_dim)
    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # (1, 1)
    mean_pooled = sum_embeddings / sum_mask  # (1, hidden_dim)
    
    # Normalize the embedding for cosine similarity
    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
    return normalized.squeeze(0)  # (hidden_dim,)


def generate_ngrams(tokens, n):
    """Generate list of n-grams from token list."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def semantic_precision(candidate, reference, n=5):
    """Computes semantic n-gram precision Pn."""
    cand_tokens = candidate.split()
    ref_tokens = reference.split()

    cand_ngrams = generate_ngrams(cand_tokens, n)
    ref_ngrams  = generate_ngrams(ref_tokens, n)

    if len(cand_ngrams) == 0 or len(ref_ngrams) == 0:
        return 0.0

    # Precompute reference embeddings
    ref_embs = [get_embedding(r) for r in ref_ngrams]

    scores = []
    for c in cand_ngrams:
        c_emb = get_embedding(c)

        # Since embeddings are normalized, cosine similarity = dot product
        sims = [
            torch.dot(c_emb, r_emb).item()
            for r_emb in ref_embs
        ]

        scores.append(max(sims))  # best match
    
    return np.mean(scores) if scores else 0.0


def bert_bleu(candidate, reference, N=4, eps=1e-8):
    """Compute BERT-BLEU-N according to your formula."""
    P = []

    for n in range(1, N+1):
        Pn = semantic_precision(candidate, reference, n)
        P.append(Pn)

    # BERT-BLEU formula
    return np.exp( (1/N) * np.sum([np.log(p + eps) for p in P]) )


# ------------------------
# Example Usage
# ------------------------

# candidate = "Ampere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie Ampère"
# reference = "Ampere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie AmpèreAmpere is the codename for a graphics processing unit (GPU) microarchitecture developed by Nvidia as the successor to both the Volta and Turing architectures. It was officially announced on May 14, 2020, and is named after French mathematician and physicist André-Marie Ampère"

# score = bert_bleu(candidate, reference)
# print("BERT-BLEU4 Score:", score)'


def main():
        
    while(True):
        
        
        candidate = input("Enter candidate sentence: ")

        reference = input("Enter reference sentence: ")
        
        score = bert_bleu(candidate, reference)
        print("BERT-BLEU4 Score:", score)

if __name__ == "__main__":
    main()