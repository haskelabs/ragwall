#!/usr/bin/env python3
"""Minimal HRCR test without heavy dependencies"""
import json
import numpy as np
from collections import Counter

# Simple cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# Fake embedder using bag-of-words
class SimpleEmbedder:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, texts):
        # Build vocabulary
        doc_freq = Counter()
        for text in texts:
            words = set(text.lower().split())
            for w in words:
                doc_freq[w] += 1

        # Assign indices and compute IDF
        for i, (word, freq) in enumerate(doc_freq.most_common(5000)):
            self.vocab[word] = i
            self.idf[word] = np.log(len(texts) / (1 + freq))

    def encode(self, texts):
        vectors = []
        for text in texts:
            vec = np.zeros(len(self.vocab))
            words = text.lower().split()
            for w in words:
                if w in self.vocab:
                    vec[self.vocab[w]] = self.idf.get(w, 1.0)
            vectors.append(vec)
        return np.array(vectors)

print("Loading data...")
corpus = [json.loads(l) for l in open('data/health_eval_corpus_mixed.jsonl')][:300]
queries = [json.loads(l) for l in open('data/health_eval_queries.jsonl')][:50]
sanitized = [json.loads(l) for l in open('data/health_eval_queries_san.jsonl')][:50]

print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}")

# Build embedder
print("Building embeddings...")
embedder = SimpleEmbedder()
corpus_texts = [d.get('text', '') for d in corpus]
embedder.fit(corpus_texts)

# Encode corpus
corpus_vecs = embedder.encode(corpus_texts)

# Find malicious docs
mal_ids = set(i for i, d in enumerate(corpus) if d.get('labels', {}).get('malicious', False))
print(f"Malicious docs: {len(mal_ids)}/{len(corpus)}")

# Evaluate
k = 5
hrcr_baseline = []
hrcr_sanitized = []
jaccard_scores = []

for q_orig, q_san in zip(queries, sanitized):
    is_attacked = q_orig.get('label') == 'attacked'

    # Baseline query
    q_text = q_orig['query']
    q_vec = embedder.encode([q_text])[0]
    sims = [cosine_sim(q_vec, c_vec) for c_vec in corpus_vecs]
    top_k_baseline = np.argsort(sims)[-k:][::-1]

    # Sanitized query
    san_text = q_san.get('sanitized', q_text)
    san_vec = embedder.encode([san_text])[0]
    sims_san = [cosine_sim(san_vec, c_vec) for c_vec in corpus_vecs]
    top_k_san = np.argsort(sims_san)[-k:][::-1]

    # HRCR for attacked queries
    if is_attacked:
        has_mal_baseline = any(i in mal_ids for i in top_k_baseline)
        has_mal_san = any(i in mal_ids for i in top_k_san)
        hrcr_baseline.append(has_mal_baseline)
        hrcr_sanitized.append(has_mal_san)

    # Jaccard for benign queries
    if not is_attacked:
        intersection = len(set(top_k_baseline) & set(top_k_san))
        jaccard = intersection / k
        jaccard_scores.append(jaccard)

# Results
hrcr_base = sum(hrcr_baseline) / len(hrcr_baseline) if hrcr_baseline else 0
hrcr_san = sum(hrcr_sanitized) / len(hrcr_sanitized) if hrcr_sanitized else 0
jaccard_mean = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

print("\n=== HEALTHCARE HRCR RESULTS ===")
print(f"Attacked queries: {len(hrcr_baseline)}")
print(f"Benign queries: {len(jaccard_scores)}")
print(f"\nHRCR@{k} (Harmful Retrieval Containment Rate):")
print(f"  Baseline: {hrcr_base:.3f}")
print(f"  Sanitized: {hrcr_san:.3f}")
print(f"  Reduction: {(hrcr_base - hrcr_san):.3f} absolute")
print(f"  Reduction: {(hrcr_base - hrcr_san)/hrcr_base if hrcr_base > 0 else 0:.1%} relative")
print(f"\nBenign Jaccard@{k}:")
print(f"  Mean: {jaccard_mean:.3f}")
print(f"  Drift: {1 - jaccard_mean:.3f}")

# Save results
results = {
    "N_queries": len(queries),
    "N_attacked": len(hrcr_baseline),
    "N_benign": len(jaccard_scores),
    "N_corpus": len(corpus),
    "N_malicious_docs": len(mal_ids),
    "HRCR@5": {
        "baseline": round(hrcr_base, 3),
        "sanitized": round(hrcr_san, 3),
        "absolute_drop": round(hrcr_base - hrcr_san, 3),
        "relative_drop": round((hrcr_base - hrcr_san)/hrcr_base if hrcr_base > 0 else 0, 3)
    },
    "Benign_Jaccard@5": {
        "mean": round(jaccard_mean, 3),
        "drift": round(1 - jaccard_mean, 3)
    }
}

import os
os.makedirs('reports/health_minimal', exist_ok=True)
with open('reports/health_minimal/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to reports/health_minimal/results.json")