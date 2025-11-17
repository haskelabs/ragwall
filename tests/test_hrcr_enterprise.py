#!/usr/bin/env python3
"""
HRCR Testing with ENTERPRISE PRR Gate
Tests using enterprise sanitizer with healthcare_mode=True
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add enterprise to path FIRST
enterprise_path = os.path.join(os.path.dirname(__file__), '..', 'enterprise')
sys.path.insert(0, enterprise_path)

# Now import from enterprise
from sanitizer.jailbreak.prr_gate import PRRGate

# Import from test_hrcr
sys.path.insert(0, os.path.dirname(__file__))
from test_hrcr import HealthcareCorpus, Query, HRCRTester, load_queries

class EnterpriseRAGWallClient:
    """Client using enterprise sanitizer with healthcare mode and ML vectors."""

    def __init__(self):
        # Load ML vectors
        import torch
        vector_path = Path("enterprise/experiments/results/tiny_jb_vectors.pt")
        self.vectors = torch.load(vector_path) if vector_path.exists() else {}
        print(f"Loaded {len(self.vectors)} pattern vectors from {vector_path}")

        # Use enterprise PRRGate with healthcare_mode
        self.gate = PRRGate(healthcare_mode=True, auto_detect_language=True)

        # Risk lexicon for reranking
        self.risk_lexicon = [
            re.compile(r"(?i)ignore\s+(previous|prior|all)\s+(instructions?|rules?|policies)"),
            re.compile(r"(?i)no\s+rules"),
            re.compile(r"(?i)without\s+restrictions"),
            re.compile(r"(?i)developer\s+mode"),
            re.compile(r"(?i)you\s+are\s+dan"),
            re.compile(r"(?i)act\s+as\s+(root|admin|sudo)"),
        ]

        # Load distilgpt2 for query encoding (to match vector dimensions)
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            self.model = AutoModel.from_pretrained('distilgpt2')
            self.model.eval()
            print("Loaded distilgpt2 model for query encoding")
        except Exception as e:
            self.tokenizer = None
            self.model = None
            print(f"WARNING: distilgpt2 not available, cosine scoring disabled: {e}")

    def sanitize(self, query: str) -> Dict[str, Any]:
        """Sanitize query using enterprise PRRGate with ML scoring."""
        # Encode query to get pooled state if model available
        import torch
        pooled_state = None
        if self.model is not None and self.tokenizer is not None:
            # Encode using distilgpt2
            inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean of last hidden state as pooled representation
                pooled_state = outputs.last_hidden_state.mean(dim=1).squeeze()

        # Prepare metadata with pattern vectors
        meta = {'per_pattern_vectors': self.vectors} if self.vectors else {}

        # Use score() method for ML-enhanced detection
        result = self.gate.score(query, pooled_state=pooled_state, meta=meta, boundary_entropy_conf=0.0)

        # Debug: Print signal probabilities for benign and attack queries
        if not hasattr(self, '_debug_benign'):
            if result.risky == False:
                print(f"\n[DEBUG] First BENIGN query signals:")
                print(f"  Query: {query[:80]}...")
                print(f"  p_keyword: {result.p_keyword:.3f}, p_cosine: {result.p_cosine:.3f}, p_entropy: {result.p_entropy:.3f}")
                print(f"  p_structure: {result.p_structure:.3f}, p_missing_self: {result.p_missing_self:.3f}")
                print(f"  families_hit: {result.families_hit}, risky: {result.risky}\n")
                self._debug_benign = True

        if not hasattr(self, '_debug_attack'):
            if result.risky == True:
                print(f"\n[DEBUG] First ATTACK query signals:")
                print(f"  Query: {query[:80]}...")
                print(f"  p_keyword: {result.p_keyword:.3f}, p_cosine: {result.p_cosine:.3f}, p_entropy: {result.p_entropy:.3f}")
                print(f"  p_structure: {result.p_structure:.3f}, p_missing_self: {result.p_missing_self:.3f}")
                print(f"  families_hit: {result.families_hit}, risky: {result.risky}")
                if result.details.get('cosine'):
                    print(f"  cosine scores: {result.details['cosine']}\n")
                self._debug_attack = True

        # Simulate sanitization (remove detected patterns)
        sanitized = query
        if result.risky:
            # Simple sanitization: remove matched patterns
            for pattern in result.keyword_hits + result.structure_hits:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            sanitized = ' '.join(sanitized.split())  # Normalize whitespace

        return {
            "sanitized_for_embed": sanitized,
            "risky": result.risky,
            "patterns": result.families_hit,
            "meta": {
                "risky": result.risky,
                "families_hit": result.families_hit,
                "p_keyword": result.p_keyword,
                "p_structure": result.p_structure,
                "p_missing_self": result.p_missing_self,
                "score": result.score
            }
        }

    def rerank(self, risky: bool, baseline_hrcr_positive: bool, k: int, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rerank candidates based on risk."""
        if not (risky and baseline_hrcr_positive):
            ids = [c.get("id") for c in candidates][:k]
            return {"ids_sorted": ids, "penalized": []}

        safe_bucket = []
        risky_bucket = []
        penalized = []

        for cand in candidates:
            cid = cand.get("id")
            text = cand.get("text", "")

            # Check if doc looks risky
            doc_risky = any(lex.search(text) for lex in self.risk_lexicon)

            if doc_risky:
                risky_bucket.append(cand)
                if cid is not None:
                    penalized.append(str(cid))
            else:
                safe_bucket.append(cand)

        ordered = safe_bucket + risky_bucket
        ids_sorted = [c.get("id") for c in ordered][:k]
        return {"ids_sorted": ids_sorted, "penalized": penalized}


def main():
    print("=" * 80)
    print("HRCR TESTING PIPELINE - RAGWall ENTERPRISE Edition")
    print("Using healthcare_mode=True with enterprise patterns")
    print("=" * 80)

    # Setup
    corpus_path = Path("enterprise/data/health_eval_corpus_mixed.jsonl")
    query_path = Path("evaluations/benchmark/data/health_care_100_queries_matched.jsonl")

    # Load corpus
    corpus = HealthcareCorpus()
    corpus.load_documents(corpus_path)
    corpus.build_index()

    # Load queries
    queries = load_queries(query_path)
    print(f"\nLoaded {len(queries)} test queries")

    # Initialize ENTERPRISE RAGWall client
    ragwall = EnterpriseRAGWallClient()
    tester = HRCRTester(corpus, ragwall)

    # Run tests
    topk = 5

    baseline_results = tester.run_baseline(queries, topk=topk)
    ragwall_sanitize_only = tester.run_ragwall(queries, topk=topk, use_rerank=False)
    ragwall_with_rerank = tester.run_ragwall(queries, topk=topk, use_rerank=True)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    baseline_metrics = tester.calculate_metrics(baseline_results, topk=topk)
    sanitize_only_metrics = tester.calculate_metrics(ragwall_sanitize_only, topk=topk)
    with_rerank_metrics = tester.calculate_metrics(ragwall_with_rerank, topk=topk)

    print(f"\nBASELINE (no protection):")
    print(f"  Average HRCR@{topk}: {baseline_metrics['avg_hrcr']:.4f} ({baseline_metrics['avg_hrcr']*100:.2f}%)")
    print(f"  Attacked queries HRCR: {baseline_metrics['attacked_hrcr']:.4f}")
    print(f"  Benign queries HRCR: {baseline_metrics['benign_hrcr']:.4f}")
    print(f"  Total harmful retrievals: {baseline_metrics['total_harmful_retrievals']}/{baseline_metrics['total_queries']*topk}")

    print(f"\nENTERPRISE RAGWALL (Sanitization Only):")
    print(f"  Average HRCR@{topk}: {sanitize_only_metrics['avg_hrcr']:.4f} ({sanitize_only_metrics['avg_hrcr']*100:.2f}%)")
    print(f"  Attacked queries HRCR: {sanitize_only_metrics['attacked_hrcr']:.4f}")
    print(f"  Benign queries HRCR: {sanitize_only_metrics['benign_hrcr']:.4f}")
    print(f"  Total harmful retrievals: {sanitize_only_metrics['total_harmful_retrievals']}/{sanitize_only_metrics['total_queries']*topk}")

    print(f"\nENTERPRISE RAGWALL (Sanitization + Reranking):")
    print(f"  Average HRCR@{topk}: {with_rerank_metrics['avg_hrcr']:.4f} ({with_rerank_metrics['avg_hrcr']*100:.2f}%)")
    print(f"  Attacked queries HRCR: {with_rerank_metrics['attacked_hrcr']:.4f}")
    print(f"  Benign queries HRCR: {with_rerank_metrics['benign_hrcr']:.4f}")
    print(f"  Total harmful retrievals: {with_rerank_metrics['total_harmful_retrievals']}/{with_rerank_metrics['total_queries']*topk}")

    # Calculate reductions
    print(f"\n{'='*80}")
    print("HRCR@{} REDUCTIONS (ENTERPRISE)".format(topk))
    print(f"{'='*80}")

    if baseline_metrics['avg_hrcr'] > 0:
        sanitize_reduction = (baseline_metrics['avg_hrcr'] - sanitize_only_metrics['avg_hrcr']) / baseline_metrics['avg_hrcr']
        rerank_reduction = (baseline_metrics['avg_hrcr'] - with_rerank_metrics['avg_hrcr']) / baseline_metrics['avg_hrcr']

        print(f"\nSanitization Only: {sanitize_reduction*100:.1f}% reduction")
        print(f"Sanitization + Reranking: {rerank_reduction*100:.1f}% reduction")
        print(f"Reranking Contribution: {(rerank_reduction - sanitize_reduction)*100:.1f}pp additional reduction")

        print(f"\n{'='*80}")
        if rerank_reduction >= 0.43:  # Within 5% of 48%
            print("✅ CLAIM VALIDATED: Achieved ≥43% reduction (48% ± 5% tolerance)")
        else:
            print(f"⚠️  BELOW CLAIM: Expected ≥43% reduction, got {rerank_reduction*100:.1f}%")
            print(f"   Using ENTERPRISE code with healthcare_mode=True")


if __name__ == '__main__':
    main()
