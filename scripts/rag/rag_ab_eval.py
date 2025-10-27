#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time, math, hashlib, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def p95(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(math.ceil(0.95 * len(ys))) - 1
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


class LMEmbedder:
    def __init__(self, model_name: str = 'gpt2-medium', device: str | None = None):
        if AutoModelForCausalLM is None:
            raise SystemExit('Please: pip install transformers torch')
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tok.pad_token is None:
            if getattr(self.tok, 'eos_token', None) is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({'pad_token': '[PAD]'})
                try:
                    self.model.resize_token_embeddings(len(self.tok))
                except Exception:
                    pass
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        self.model.to(self.device); self.model.eval()
        with torch.no_grad():
            dummy = self.tok('ok', return_tensors='pt')
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            _ = self.model(**dummy)

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16, pool: str = 'mean') -> torch.Tensor:
        vecs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = self.tok(chunk, return_tensors='pt', padding=True, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1]  # [B,T,D]
            am = enc['attention_mask'].unsqueeze(-1)  # [B,T,1]
            if pool == 'mean':
                x = (hs * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)
            else:
                x = hs[:, -1, :]
            x = torch.nn.functional.normalize(x, p=2, dim=-1).detach().cpu()
            vecs.append(x)
        return torch.cat(vecs, dim=0)


class STEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str | None = None):
        if SentenceTransformer is None:
            raise SystemExit('Please: pip install sentence-transformers')
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)
        # warmup
        _ = self.model.encode(['ok'], convert_to_tensor=True, normalize_embeddings=True)

    def embed(self, texts: List[str], batch_size: int = 32, pool: str = 'mean') -> torch.Tensor:
        embs = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        return embs.detach().cpu()


def topk_indices(sim: torch.Tensor, k: int) -> torch.Tensor:
    # sim: [Q, D]
    kv = min(k, sim.size(1))
    _, idx = torch.topk(sim, kv, dim=1, largest=True, sorted=True)
    return idx


def cosine_sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [N,D]; B: [M,D] ; expects normalized
    return torch.matmul(A, B.t())


def jaccard_at_k(ids_a: List[str], ids_b: List[str], k: int) -> float:
    sa = set(ids_a[:k]); sb = set(ids_b[:k])
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    if uni == 0:
        return 1.0
    return float(inter / uni)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2-medium')
    ap.add_argument('--embedder', default='hf', choices=['hf','st'],
                    help='Embedding backend: hf (LM mean-pool) or st (sentence-transformers)')
    ap.add_argument('--st-model', default='all-MiniLM-L6-v2',
                    help='Sentence-Transformers model when --embedder st')
    ap.add_argument('--corpus', required=True, help='JSONL with {id, text, labels.malicious: bool}')
    ap.add_argument('--queries', required=True, help='Original queries JSONL: {query, pattern?}')
    ap.add_argument('--sanitized', required=True, help='Sanitized queries JSONL from sanitize_queries.py')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--k2', type=int, default=0, help='Optional secondary k (e.g., 10) for HRCR@k2 summary')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--steer-alpha', type=float, default=0.0,
                    help='Directional steering away from malicious centroid for sanitized queries (0.0 disables)')
    ap.add_argument('--penalty', type=float, default=0.1,
                    help='Re-rank demotion weight for risky docs when --rerank-risk is set')
    ap.add_argument('--rerank-risk', action='store_true',
                    help='Apply a tiny risk-aware demotion to sanitized candidates matching injection lexicon')
    ap.add_argument('--soft-proto-reject', action='store_true',
                    help='Penalize similarity to a jailbreak prototype even without lexicon hits (doc-side)')
    ap.add_argument('--proto-weight', type=float, default=0.0,
                    help='Prototype penalty weight (0 disables). Used only when --soft-proto-reject')
    ap.add_argument('--bootstrap', type=int, default=0, help='Bootstrap samples for CIs (0 disables)')
    ap.add_argument('--doc-embeddings', default=None,
                    help='Optional path to load/store precomputed corpus embeddings (.pt).')
    ap.add_argument('--baseline-embeddings', default=None,
                    help='Optional path for baseline query embeddings cache (.pt).')
    ap.add_argument('--sanitized-embeddings', default=None,
                    help='Optional path for sanitized query embeddings cache (.pt).')
    ap.add_argument('--cache-readonly', action='store_true',
                    help='Only read from caches; do not write when missing.')
    args = ap.parse_args()

    # Lock determinism for tie-breaks and any stochastic ops
    try:
        import numpy as _np
        _np.random.seed(42)
    except Exception:
        _np = None  # type: ignore
    random.seed(42)
    if torch is not None:
        try:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            if getattr(torch.backends, 'cudnn', None) is not None:
                torch.backends.cudnn.deterministic = True  # type: ignore
                torch.backends.cudnn.benchmark = False  # type: ignore
        except Exception:
            pass

    # Load corpus
    corp = load_jsonl(args.corpus)
    if not corp:
        raise SystemExit('Empty corpus')

    docs = [str(r.get('text', '')).strip() for r in corp]
    if any(len(t) == 0 for t in docs):
        raise SystemExit('Corpus contains empty/missing text fields')
    doc_ids = [str(r.get('id', i)) for i, r in enumerate(corp)]
    assert len(doc_ids) == len(set(doc_ids)), 'Duplicate doc_ids in corpus'

    # Read labels correctly (labels.malicious OR top-level malicious)
    labels_list = [(r.get('labels') or {}) for r in corp]
    is_mal: List[bool] = [
        bool(r.get('malicious', False) or lab.get('malicious', False))
        for r, lab in zip(corp, labels_list)
    ]
    # Optional: capture corpus-side pattern if present (not currently used in metrics)
    doc_pattern: List[str | None] = [
        (lab.get('pattern') if isinstance(lab.get('pattern'), str) else (r.get('pattern') if isinstance(r.get('pattern'), str) else None))
        for r, lab in zip(corp, labels_list)
    ]

    # Load queries (keep rows for per-pattern breakdown)
    base_rows = load_jsonl(args.queries)
    q_base = [str(r.get('query', '')).strip() for r in base_rows]
    san_rows = load_jsonl(args.sanitized)
    q_san = [str(r.get('sanitized', '')).strip() for r in san_rows]
    risky_flags = [bool((r.get('meta') or {}).get('risky', False)) for r in san_rows]

    # Align lengths
    n = min(len(q_base), len(q_san), len(risky_flags))
    q_base = q_base[:n]; q_san = q_san[:n]; risky_flags = risky_flags[:n]
    base_rows = base_rows[:n]; san_rows = san_rows[:n]

    # Optional per-query pattern (prefer explicit; then sanitized row; then meta.pattern). Fallback to pattern heuristics.
    def infer_pattern_from_text(q: str) -> str:
        t = (q or '').lower()
        rules = [
            ('ignore', ['ignore previous', 'disregard', 'without restrictions', 'no rules']),
            ('developer_mode', ['developer mode', 'dev mode']),
            ('dan', [' you are dan', ' as dan', ' dan']),
            ('as_root', [' as root', ' sudo ']),
            ('role_play', ['role-play', 'role play', 'act as']),
        ]
        for label, keys in rules:
            if any(k in t for k in keys):
                return label
        return 'benign'
    q_patterns: List[str] = []
    for i in range(n):
        meta_s = san_rows[i].get('meta') or {}
        pat = (
            base_rows[i].get('pattern')
            or san_rows[i].get('pattern')
            or meta_s.get('pattern')
            or ''
        )
        if not pat:
            # Tiny fallback to keep per-pattern HRCR populated
            pat = infer_pattern_from_text(base_rows[i].get('query',''))
        q_patterns.append(str(pat))

    # Ensure k <= number of documents
    k_req = int(args.k)
    if len(docs) == 0:
        raise SystemExit('Corpus has zero docs')
    if k_req > len(docs):
        print(f'[warn] Requested k={k_req} > corpus size {len(docs)}; using k={len(docs)}')
        k_req = len(docs)

    os.makedirs(args.outdir, exist_ok=True)

    cache_readonly = bool(args.cache_readonly)

    # Choose embedder
    if args.embedder == 'st':
        emb = STEmbedder(model_name=args.st_model)
        # Optional device guard: if CUDA is available, ensure ST runs on CUDA
        if torch is not None and torch.cuda.is_available():
            try:
                assert str(emb.model.device).startswith('cuda'), f"ST model device is {emb.model.device}, expected CUDA when available"
            except Exception:
                pass
        bs_docs = max(32, int(args.batch_size))
    else:
        emb = LMEmbedder(model_name=args.model)
        bs_docs = int(args.batch_size)

    def ensure_parent(path_str: Optional[str]) -> None:
        if not path_str:
            return
        parent = Path(path_str).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    def load_cached(path_str: Optional[str]) -> Tuple[Optional['torch.Tensor'], Dict[str, Any]]:
        if not path_str:
            return None, {}
        p = Path(path_str).expanduser()
        if not p.exists():
            return None, {}
        if torch is None:
            raise SystemExit('torch is required to load cached embeddings')
        data = torch.load(str(p), map_location='cpu')
        if isinstance(data, dict) and 'embeddings' in data:
            tensor = data['embeddings']
            meta = data.get('meta') or {}
        elif isinstance(data, torch.Tensor):
            tensor = data
            meta = {}
        else:
            raise SystemExit(f'Unrecognized cache format at {p}')
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f'Cache at {p} does not contain a tensor')
        return tensor, meta

    def validate_cache(meta: Dict[str, Any], expected: Dict[str, Any], label: str, path_str: str) -> None:
        if not expected:
            return
        missing = [k for k in expected if meta.get(k) != expected[k]]
        if missing:
            details = ', '.join(f'{k}={meta.get(k)!r} (expected {expected[k]!r})' for k in missing)
            raise SystemExit(f'{label} cache mismatch at {path_str}: {details}')

    def save_cache(path_str: Optional[str], tensor: 'torch.Tensor', meta: Dict[str, Any]) -> None:
        if not path_str:
            return
        if torch is None:
            raise SystemExit('torch is required to persist cached embeddings')
        ensure_parent(path_str)
        payload = {'embeddings': tensor.cpu(), 'meta': meta}
        torch.save(payload, str(Path(path_str).expanduser()))

    doc_sha = file_sha256(args.corpus)
    base_sha = file_sha256(args.queries)
    san_sha = file_sha256(args.sanitized)

    # Embed docs and queries
    D_meta_expected = {
        'kind': 'docs',
        'backend': 'st' if args.embedder == 'st' else 'hf',
        'model': args.st_model if args.embedder == 'st' else args.model,
        'source_sha': doc_sha,
        'count': len(docs),
    }
    D_cached, D_meta = load_cached(args.doc_embeddings)
    if D_cached is not None:
        validate_cache(D_meta, D_meta_expected, 'Doc embeddings', args.doc_embeddings)
        D = D_cached
    else:
        D = emb.embed(docs, batch_size=bs_docs)  # [Nd, d]
        if not cache_readonly:
            save_cache(args.doc_embeddings, D, D_meta_expected)
    if D.size(0) != len(docs):
        raise SystemExit(f'Doc embeddings row mismatch: expected {len(docs)}, found {int(D.size(0))}')

    # Normalization invariant (cosine expects normalized vectors)
    try:
        assert abs((D**2).sum(dim=1).mean().item() - 1.0) < 1e-3
    except Exception:
        pass
    Qb_meta_expected = {
        'kind': 'baseline_queries',
        'backend': 'st' if args.embedder == 'st' else 'hf',
        'model': args.st_model if args.embedder == 'st' else args.model,
        'source_sha': base_sha,
        'count': len(q_base),
    }
    Qb_cached, Qb_meta = load_cached(args.baseline_embeddings)
    if Qb_cached is not None:
        validate_cache(Qb_meta, Qb_meta_expected, 'Baseline query embeddings', args.baseline_embeddings)
        Qb = Qb_cached
    else:
        Qb = emb.embed(q_base, batch_size=int(args.batch_size))  # [Nq, d]
        if not cache_readonly:
            save_cache(args.baseline_embeddings, Qb, Qb_meta_expected)
    if Qb.size(0) != len(q_base):
        raise SystemExit(f'Baseline embedding row mismatch: expected {len(q_base)}, found {int(Qb.size(0))}')

    # Build sanitized embeddings with zero-drift reuse where identical
    lat_ms: List[float] = []
    Qs_meta_expected = {
        'kind': 'sanitized_queries',
        'backend': 'st' if args.embedder == 'st' else 'hf',
        'model': args.st_model if args.embedder == 'st' else args.model,
        'source_sha': san_sha,
        'count': len(q_san),
    }
    Qs_cached, Qs_meta = load_cached(args.sanitized_embeddings)
    if Qs_cached is not None:
        validate_cache(Qs_meta, Qs_meta_expected, 'Sanitized query embeddings', args.sanitized_embeddings)
        Qs = Qs_cached
        lat_ms = [0.0]
    else:
        Qs_blocks: List[torch.Tensor] = []
        bs = int(args.batch_size)
        for i in range(0, n, bs):
            base_chunk = q_base[i:i+bs]
            san_chunk = q_san[i:i+bs]

            # If the entire chunk is identical, reuse embeddings (zero latency)
            if base_chunk == san_chunk:
                Qs_blocks.append(Qb[i:i+bs])
                lat_ms.append(0.0)
                continue

            # Mixed case: reuse per-item when identical; embed only the changed ones
            eq_mask = [int(b == s) for b, s in zip(base_chunk, san_chunk)]
            need_embed_idx = [j for j, m in enumerate(eq_mask) if m == 0]

            if len(need_embed_idx) == 0:
                Qs_blocks.append(Qb[i:i+bs])
                lat_ms.append(0.0)
            else:
                texts_to_embed = [san_chunk[j] for j, flag in enumerate(eq_mask) if flag == 0]
                t0 = time.time()
                V_part = emb.embed(texts_to_embed, batch_size=int(args.batch_size))
                lat_ms.append((time.time() - t0) * 1000.0)
                # stitch back together
                V_full = Qb[i:i+bs].clone()
                ptr = 0
                for j, flag in enumerate(eq_mask):
                    if flag == 0:
                        V_full[j:j+1] = V_part[ptr:ptr+1]
                        ptr += 1
                Qs_blocks.append(V_full)

        Qs = torch.cat(Qs_blocks, dim=0)
        if not cache_readonly:
            save_cache(args.sanitized_embeddings, Qs, Qs_meta_expected)
    if Qs.size(0) != len(q_san):
        raise SystemExit(f'Sanitized embedding row mismatch: expected {len(q_san)}, found {int(Qs.size(0))}')
    if D.size(1) != Qb.size(1) or D.size(1) != Qs.size(1):
        raise SystemExit('Embedding dimension mismatch between corpus, baseline queries, and sanitized queries')

    # Deterministic tie-breaks: add tiny increasing epsilon by doc index to both arms
    eps = 1e-9
    doc_rank = torch.arange(D.size(0), device=D.device, dtype=D.dtype)  # [Nd]

    # Baseline similarities/top-k
    sim_b = cosine_sim(Qb, D) + eps * doc_rank.unsqueeze(0)
    idx_b = topk_indices(sim_b, k_req)  # [N, k]

    # Count malicious in baseline top-k using indices directly (robust to duplicate ids)
    is_mal_tensor = torch.tensor(is_mal, dtype=torch.bool, device=sim_b.device)
    baseline_mal = [
        int(is_mal_tensor[idx_b[i][:k_req]].sum().item())
        for i in range(n)
    ]

    # Build mask: apply adjustments only when the query is risky AND baseline top-k contains malicious
    q_mask_hr = torch.tensor(
        [1.0 if (bool(risky_flags[i]) and (baseline_mal[i] > 0)) else 0.0 for i in range(n)],
        dtype=D.dtype, device=D.device
    ).unsqueeze(1)  # [N,1]

    # Optional directional steering for sanitized queries away from malicious centroid (only for masked)
    if args.steer_alpha and args.steer_alpha > 0.0:
        mal_mask = is_mal_tensor
        if mal_mask.any() and (~mal_mask).any():
            m_cent = D[mal_mask].mean(dim=0)
            b_cent = D[~mal_mask].mean(dim=0)
            m_dir = m_cent - b_cent
            m_dir = m_dir / (m_dir.norm(p=2) + 1e-9)
            proj = (Qs @ m_dir.unsqueeze(1)) * m_dir.unsqueeze(0)  # [N,d]
            Qs = Qs - q_mask_hr * proj * float(args.steer_alpha)
            Qs = Qs / (Qs.norm(p=2, dim=-1, keepdim=True) + 1e-9)

    # Sanitized similarities/top-k (+ deterministic epsilon)
    sim_s = cosine_sim(Qs, D) + eps * doc_rank.unsqueeze(0)

    # Optional risk-aware re-rank: per-doc penalty, applied only to masked queries
    if args.rerank_risk:
        lex = [
            'ignore previous', 'disregard', 'no rules', 'without restrictions',
            'developer mode', 'act as', 'you are dan', 'printenv', 'dump all', 'base64',
        ]
        doc_pen = torch.tensor(
            [float(args.penalty) if any(x in str(t).lower() for x in lex) else 0.0 for t in docs],
            dtype=sim_s.dtype, device=sim_s.device
        )  # [Nd]
        sim_s = sim_s - q_mask_hr * doc_pen.unsqueeze(0)

    # Optional soft semantic prototype rejection (doc-side cosine to prototype)
    if args.soft_proto_reject and (args.proto_weight or 0.0) > 0.0:
        # Build prototype from docs matching jailbreak cues; fallback to malicious docs
        jb_idx: List[int] = []
        try:
            from rag_sanitizer.patterns import PATTERN_KEYWORDS as _PK
            from rag_sanitizer.patterns import STRUCTURE_PATTERNS as _SP  # type: ignore
            for j, t in enumerate(docs):
                tl = str(t).lower()
                if any(any(kw in tl for kw in kws) for kws in _PK.values()):
                    jb_idx.append(j)
                    continue
                if any(rx.search(tl) for rx in _SP):
                    jb_idx.append(j)
        except Exception:
            pass
        if not jb_idx:
            jb_idx = [j for j, m in enumerate(is_mal) if m]
        if jb_idx:
            P = D[jb_idx].mean(dim=0)
            P = P / (P.norm(p=2) + 1e-9)
            doc_proto = torch.clamp(torch.mv(D, P), min=0.0)  # [Nd], relu(cos)
            sim_s = sim_s - q_mask_hr * (float(args.proto_weight) * doc_proto.unsqueeze(0))

    idx_s = topk_indices(sim_s, k_req)

    # Optional secondary k2 for HRCR@k2
    k2 = int(args.k2) if getattr(args, 'k2', 0) else 0
    if k2 > 0 and k2 != k_req:
        if k2 > len(docs):
            print(f'[warn] Requested k2={k2} > corpus size {len(docs)}; using k2={len(docs)}')
            k2 = len(docs)
        idx_b2 = topk_indices(sim_b, k2)
        idx_s2 = topk_indices(sim_s, k2)
    else:
        k2 = 0

    # Prepare both index and id views
    topidx_b: List[List[int]] = [idx_b[i].tolist() for i in range(n)]
    topidx_s: List[List[int]] = [idx_s[i].tolist() for i in range(n)]
    topids_b: List[List[str]] = [[doc_ids[j] for j in topidx_b[i]] for i in range(n)]
    topids_s: List[List[str]] = [[doc_ids[j] for j in topidx_s[i]] for i in range(n)]

    # Metrics
    hrcr_b: List[float] = []
    hrcr_s: List[float] = []
    benign_ratio_b: List[float] = []
    benign_ratio_s: List[float] = []
    jacc_benign: List[float] = []

    for i in range(n):
        idxs_b = topidx_b[i][:k_req]
        idxs_s = topidx_s[i][:k_req]
        mal_b = int(is_mal_tensor[idxs_b].sum().item())
        mal_s = int(is_mal_tensor[idxs_s].sum().item())
        if risky_flags[i]:
            hrcr_b.append(mal_b / float(k_req))
            hrcr_s.append(mal_s / float(k_req))
            benign_ratio_b.append(1.0 - (mal_b / float(k_req)))
            benign_ratio_s.append(1.0 - (mal_s / float(k_req)))
        else:
            jacc_benign.append(jaccard_at_k(topids_b[i], topids_s[i], k_req))

    lat_core = lat_ms[1:] if len(lat_ms) > 1 else lat_ms
    summary: Dict[str, Any] = {
        'N_queries': n,
        'N_attacked': int(sum(1 for f in risky_flags if f)),
        'N_benign': int(sum(1 for f in risky_flags if not f)),
        f'HRCR@{k_req}': {
            'baseline': mean(hrcr_b),
            'sanitized': mean(hrcr_s),
            'absolute_drop': max(0.0, mean(hrcr_b) - mean(hrcr_s)) if hrcr_b else 0.0,
            'relative_drop': ((mean(hrcr_b) - mean(hrcr_s)) / max(1e-6, mean(hrcr_b))) if hrcr_b else 0.0,
        },
        f'Benign_Jaccard@{k_req}': {
            'mean': mean(jacc_benign),
            'drift': 1.0 - mean(jacc_benign),
        },
        'faithfulness_proxy_attacked': {
            'baseline_benign_ratio@k': mean(benign_ratio_b),
            'sanitized_benign_ratio@k': mean(benign_ratio_s),
            'relative_gain': ((mean(benign_ratio_s) - mean(benign_ratio_b)) / max(1e-6, mean(benign_ratio_b))) if benign_ratio_b else 0.0,
        },
        'embed_latency_ms': {
            'batches': lat_ms,
            'p95_excl_first': p95(lat_core),
        },
        'knobs': {
            'embedder_backend': 'st' if args.embedder == 'st' else 'hf',
            'embedder_model': (args.st_model if args.embedder == 'st' else args.model),
            'steer_alpha': float(args.steer_alpha or 0.0),
            'risk_aware_rerank': bool(args.rerank_risk),
            'risk_rerank_penalty': float(args.penalty if args.rerank_risk else 0.0),
            'soft_proto_reject': bool(args.soft_proto_reject),
            'proto_weight': float(args.proto_weight or 0.0),
            'mask_condition': 'risky && baseline_mal>0',
            'deterministic_tie_breaker': 'doc_index_epsilon',
            'seed': 42,
            'bootstrap_samples': int(args.bootstrap or 0),
            # If sanitize pipeline provided a separate generation-only string, record presence
            'suffix_for_generation_only': any(bool((r.get('sanitized_for_gen') or '') and (r.get('sanitized_for_gen') != r.get('sanitized'))) for r in san_rows),
            # Not applicable in this script but useful for docs of upstream guard
            'regex_cosine_gate_active': False,
        },
    }

    # If k2 requested, compute HRCR@k2 aggregates for attacked queries
    if k2 > 0:
        hrcr_b2: List[float] = []
        hrcr_s2: List[float] = []
        for i in range(n):
            if risky_flags[i]:
                mal_b2 = int(is_mal_tensor[idx_b2[i][:k2]].sum().item()) / float(k2)
                mal_s2 = int(is_mal_tensor[idx_s2[i][:k2]].sum().item()) / float(k2)
                hrcr_b2.append(mal_b2)
                hrcr_s2.append(mal_s2)
        summary[f'HRCR@{k2}'] = {
            'baseline': mean(hrcr_b2),
            'sanitized': mean(hrcr_s2),
            'absolute_drop': max(0.0, mean(hrcr_b2) - mean(hrcr_s2)) if hrcr_b2 else 0.0,
            'relative_drop': ((mean(hrcr_b2) - mean(hrcr_s2)) / max(1e-6, mean(hrcr_b2))) if hrcr_b2 else 0.0,
        }

    # Per-pattern HRCR (by query pattern) – risky only
    from collections import defaultdict
    pat_hr: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {'b': [], 's': []})
    for i in range(n):
        if risky_flags[i]:
            pat = q_patterns[i] or ''
            mal_b_i = int(is_mal_tensor[topidx_b[i][:k_req]].sum().item()) / float(k_req)
            mal_s_i = int(is_mal_tensor[topidx_s[i][:k_req]].sum().item()) / float(k_req)
            pat_hr[pat]['b'].append(mal_b_i)
            pat_hr[pat]['s'].append(mal_s_i)
    if pat_hr:
        summary['per_pattern_HRCR'] = {
            p: {
                'baseline': mean(v['b']),
                'sanitized': mean(v['s']),
                'relative_drop': (mean(v['b']) - mean(v['s'])) / max(1e-6, mean(v['b']))
            }
            for p, v in pat_hr.items()
        }
        summary['per_pattern_counts'] = {p: len(v['b']) for p, v in pat_hr.items()}

    # Per-query records
    perq: List[Dict[str, Any]] = []
    for i in range(n):
        hr_b_i = int(is_mal_tensor[topidx_b[i][:k_req]].sum().item()) / float(k_req)
        hr_s_i = int(is_mal_tensor[topidx_s[i][:k_req]].sum().item()) / float(k_req)
        perq.append({
            'query': q_base[i],
            'sanitized': q_san[i],
            'risky': bool(risky_flags[i]),
            'pattern': q_patterns[i],
            'topk_baseline': topids_b[i],
            'topk_sanitized': topids_s[i],
            f'hrcr_baseline@{k_req}': hr_b_i,
            f'hrcr_sanitized@{k_req}': hr_s_i,
            f'jaccard@{k_req}': jaccard_at_k(topids_b[i], topids_s[i], k_req),
            'original_sha256': sha256(q_base[i]),
            'sanitized_sha256': sha256(q_san[i]),
        })

    # Optional bootstrap CIs
    B = int(args.bootstrap)
    if B > 0:
        # Ensure numpy available for percentile CIs
        if _np is None:  # type: ignore[name-defined]
            try:
                import numpy as _np  # type: ignore
            except Exception:
                raise SystemExit('Bootstrap requested but numpy is not available. Please: pip install numpy')
        attacked_idx = [i for i in range(n) if risky_flags[i]]
        benign_idx = [i for i in range(n) if not risky_flags[i]]
        hr_b_samps, hr_s_samps, drifts, faith_b_samps, faith_s_samps = [], [], [], [], []
        for _ in range(B):
            if attacked_idx:
                samp_att = [random.choice(attacked_idx) for __ in range(len(attacked_idx))]
                hb = sum(int(is_mal_tensor[topidx_b[i][:k_req]].sum().item()) / float(k_req) for i in samp_att) / max(1, len(samp_att))
                hs = sum(int(is_mal_tensor[topidx_s[i][:k_req]].sum().item()) / float(k_req) for i in samp_att) / max(1, len(samp_att))
                hr_b_samps.append(hb); hr_s_samps.append(hs)
                fb = sum(1.0 - (int(is_mal_tensor[topidx_b[i][:k_req]].sum().item()) / float(k_req)) for i in samp_att) / max(1, len(samp_att))
                fs = sum(1.0 - (int(is_mal_tensor[topidx_s[i][:k_req]].sum().item()) / float(k_req)) for i in samp_att) / max(1, len(samp_att))
                faith_b_samps.append(fb); faith_s_samps.append(fs)
            if benign_idx:
                samp_ben = [random.choice(benign_idx) for __ in range(len(benign_idx))]
                drifts.append(1.0 - sum(jaccard_at_k(topids_b[i], topids_s[i], k_req) for i in samp_ben) / max(1, len(samp_ben)))

        def ci(a: List[float]):
            if not a:
                return [None, None, None]
            qs = _np.percentile(_np.array(a, dtype=float), [2.5, 50, 97.5]).tolist()
            return [float(x) for x in qs]

        summary['bootstrap'] = {
            f'HRCR@{k_req}': {
                'baseline_CI': ci(hr_b_samps),
                'sanitized_CI': ci(hr_s_samps),
                'relative_drop_CI': ci([(b - s) / max(1e-6, b) for b, s in zip(hr_b_samps, hr_s_samps)]),
            },
            f'Benign_Jaccard@{k_req}': {
                'drift_CI': ci(drifts),
            },
            'faithfulness_proxy_attacked': {
                'benign_ratio_baseline_CI': ci(faith_b_samps),
                'benign_ratio_sanitized_CI': ci(faith_s_samps),
                'relative_gain_CI': ci([(s - b) / max(1e-6, b) for b, s in zip(faith_b_samps, faith_s_samps)]),
            },
        }

    with open(os.path.join(args.outdir, 'rag_ab_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, 'rag_ab_per_query.json'), 'w') as f:
        json.dump(perq, f, indent=2)
    with open(os.path.join(args.outdir, 'rag_ab_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print('Saved', os.path.join(args.outdir, 'rag_ab_summary.json'))


if __name__ == '__main__':
    main()
