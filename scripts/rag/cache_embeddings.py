#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

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


class LMEmbedder:
    def __init__(self, model_name: str = 'gpt2-medium', device: Optional[str] = None):
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
        self.model.to(self.device)
        self.model.eval()
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
            hs = out.hidden_states[-1]
            am = enc['attention_mask'].unsqueeze(-1)
            if pool == 'mean':
                x = (hs * am).sum(dim=1) / am.sum(dim=1).clamp_min(1.0)
            else:
                x = hs[:, -1, :]
            x = torch.nn.functional.normalize(x, p=2, dim=-1).detach().cpu()
            vecs.append(x)
        return torch.cat(vecs, dim=0)


class STEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        if SentenceTransformer is None:
            raise SystemExit('Please: pip install sentence-transformers')
        self.model = SentenceTransformer(model_name)
        if device:
            self.model = self.model.to(device)
        _ = self.model.encode(['ok'], convert_to_tensor=True, normalize_embeddings=True)

    def embed(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        embs = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        return embs.detach().cpu()


def ensure_parent(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def save_tensor(path: str, tensor: torch.Tensor, meta: Dict[str, Any], overwrite: bool) -> None:
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        raise SystemExit(f'{p} exists; pass --overwrite to replace')
    ensure_parent(str(p))
    torch.save({'embeddings': tensor.cpu(), 'meta': meta}, str(p))


def load_tensor(path: str) -> torch.Tensor:
    p = Path(path).expanduser()
    data = torch.load(str(p), map_location='cpu')
    if isinstance(data, dict) and 'embeddings' in data:
        return data['embeddings']
    if isinstance(data, torch.Tensor):
        return data
    raise SystemExit(f'Unexpected tensor format at {p}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--embedder', choices=['hf', 'st'], default='hf')
    ap.add_argument('--model', default='gpt2-medium', help='LM name when --embedder hf')
    ap.add_argument('--st-model', default='all-MiniLM-L6-v2', help='Sentence-Transformers model when --embedder st')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default=None, help='Optional torch device override')
    ap.add_argument('--corpus', help='Corpus JSONL input for document embeddings')
    ap.add_argument('--doc-out', help='Output path for corpus embeddings (.pt)')
    ap.add_argument('--queries', help='Baseline queries JSONL input')
    ap.add_argument('--baseline-out', help='Output path for baseline query embeddings (.pt)')
    ap.add_argument('--sanitized', help='Sanitized queries JSONL input')
    ap.add_argument('--sanitized-out', help='Output path for sanitized query embeddings (.pt)')
    ap.add_argument('--overwrite', action='store_true', help='Allow overwriting existing cache files')
    args = ap.parse_args()

    if torch is None:
        raise SystemExit('torch is required to pre-compute embeddings')

    backend = 'st' if args.embedder == 'st' else 'hf'
    model_name = args.st_model if backend == 'st' else args.model

    if args.embedder == 'st':
        emb = STEmbedder(model_name=args.st_model, device=args.device)
        doc_bs = max(args.batch_size, 32)
    else:
        emb = LMEmbedder(model_name=args.model, device=args.device)
        doc_bs = args.batch_size

    def embed_texts(texts: List[str], batch_size: int) -> torch.Tensor:
        if not texts:
            raise SystemExit('No texts supplied for embedding')
        return emb.embed(texts, batch_size=batch_size)  # type: ignore[attr-defined]

    if args.corpus:
        if not args.doc_out:
            raise SystemExit('--doc-out required when --corpus is provided')
        corpus_rows = load_jsonl(args.corpus)
        docs = [str(r.get('text', '')).strip() for r in corpus_rows]
        if any(len(t) == 0 for t in docs):
            raise SystemExit('Corpus contains empty/missing text fields')
        doc_tensor = embed_texts(docs, max(1, doc_bs))
        meta = {
            'kind': 'docs',
            'backend': backend,
            'model': model_name,
            'source_sha': file_sha256(args.corpus),
            'count': len(docs),
        }
        save_tensor(args.doc_out, doc_tensor, meta, args.overwrite)
        print(f'[cache] Saved doc embeddings -> {args.doc_out}')

    base_rows: Optional[List[Dict[str, Any]]] = None
    base_tensor: Optional[torch.Tensor] = None
    if args.queries:
        if not args.baseline_out:
            raise SystemExit('--baseline-out required when --queries is provided')
        base_rows = load_jsonl(args.queries)
        q_base = [str(r.get('query', '')).strip() for r in base_rows]
        base_tensor = embed_texts(q_base, args.batch_size)
        meta = {
            'kind': 'baseline_queries',
            'backend': backend,
            'model': model_name,
            'source_sha': file_sha256(args.queries),
            'count': len(q_base),
        }
        save_tensor(args.baseline_out, base_tensor, meta, args.overwrite)
        print(f'[cache] Saved baseline query embeddings -> {args.baseline_out}')

    if args.sanitized:
        if not args.sanitized_out:
            raise SystemExit('--sanitized-out required when --sanitized is provided')
        if not args.queries:
            raise SystemExit('--queries required alongside --sanitized for alignment')
        san_rows = load_jsonl(args.sanitized)
        if base_rows is None:
            base_rows = load_jsonl(args.queries)
            q_base = [str(r.get('query', '')).strip() for r in base_rows]
            base_tensor = load_tensor(args.baseline_out)
        else:
            q_base = [str(r.get('query', '')).strip() for r in base_rows]
        q_san = [str(r.get('sanitized') or r.get('query') or '').strip() for r in san_rows]
        n = min(len(q_base), len(q_san))
        if n == 0:
            raise SystemExit('Sanitized file produced zero rows')
        q_base = q_base[:n]
        q_san = q_san[:n]
        base_tensor = base_tensor[:n] if isinstance(base_tensor, torch.Tensor) else None  # type: ignore[index]
        if base_tensor is not None and int(base_tensor.size(0)) != n:
            base_tensor = None
        reuse = base_tensor is not None
        if reuse:
            bs = max(1, args.batch_size)
            blocks: List[torch.Tensor] = []
            for i in range(0, n, bs):
                b_chunk = q_base[i:i+bs]
                s_chunk = q_san[i:i+bs]
                if b_chunk == s_chunk:
                    blocks.append(base_tensor[i:i+bs])
                    continue
                eq_mask = [int(b == s) for b, s in zip(b_chunk, s_chunk)]
                need_idx = [j for j, flag in enumerate(eq_mask) if flag == 0]
                if not need_idx:
                    blocks.append(base_tensor[i:i+bs])
                    continue
                texts = [s_chunk[j] for j in need_idx]
                embeds = embed_texts(texts, args.batch_size)
                stitched = base_tensor[i:i+bs].clone()
                pos = 0
                for j, flag in enumerate(eq_mask):
                    if flag == 0:
                        stitched[j:j+1] = embeds[pos:pos+1]
                        pos += 1
                blocks.append(stitched)
            san_tensor = torch.cat(blocks, dim=0)
        else:
            san_tensor = embed_texts(q_san, args.batch_size)
        meta = {
            'kind': 'sanitized_queries',
            'backend': backend,
            'model': model_name,
            'source_sha': file_sha256(args.sanitized),
            'count': n,
        }
        save_tensor(args.sanitized_out, san_tensor, meta, args.overwrite)
        print(f'[cache] Saved sanitized query embeddings -> {args.sanitized_out}')


if __name__ == '__main__':
    main()
