#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sanitizer.rag_sanitizer import SanitizerConfig, QuerySanitizer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True, help='JSONL with {query, label: attacked|benign}')
    ap.add_argument('--healthcare', action='store_true', help='Enable healthcare pattern set')
    ap.add_argument('--rules-only', action='store_true', help='Force regex/structure-only gate (no model)')
    args = ap.parse_args()

    cfg = SanitizerConfig(
        model_name='__disabled__' if args.rules_only else 'gpt2-medium',
        vectors_path='',
        layer='transformer.h.1',
        pool_k=6,
        max_edit_positions=6,
        scale=0.2,
        pattern_gate=True,
        no_overall=True,
        orthogonalize=True,
        entropy_scale=False,
        quorum=2,
        healthcare_mode=bool(args.healthcare),
    )
    san = QuerySanitizer(cfg)

    rows = load_jsonl(args.queries)
    y_true: List[int] = []  # 1=attacked, 0=benign
    y_pred: List[int] = []  # 1=risky, 0=not risky
    per_pattern: Dict[str, Dict[str, int]] = {}

    def norm_pat(p: str) -> str:
        return (p or '').strip().lower() or 'unknown'

    for r in rows:
        q = str(r.get('query', '')).strip()
        if not q:
            continue
        lab = str(r.get('label', '')).strip().lower()
        y_true.append(1 if lab == 'attacked' else 0)
        _, meta = san.sanitize_query(q)
        risky = bool((meta or {}).get('risky', False))
        y_pred.append(1 if risky else 0)
        pat = norm_pat(str(r.get('pattern', '')))
        if pat not in per_pattern:
            per_pattern[pat] = {'tp':0,'fp':0,'tn':0,'fn':0}
        if lab == 'attacked' and risky:
            per_pattern[pat]['tp'] += 1
        elif lab == 'attacked' and not risky:
            per_pattern[pat]['fn'] += 1
        elif lab != 'attacked' and risky:
            per_pattern[pat]['fp'] += 1
        else:
            per_pattern[pat]['tn'] += 1

    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)

    prec = float(tp/(tp+fp)) if (tp+fp)>0 else 0.0
    rec = float(tp/(tp+fn)) if (tp+fn)>0 else 0.0
    det_rate = float(tp+fn and (tp+fn) and tp/(tp+fn))  # among attacked only

    print(f"Detected attacked: {tp}/{tp+fn} ({(tp/(tp+fn)*100 if (tp+fn)>0 else 0):.1f}%)")
    print(f"Precision: {(prec*100):.1f}%  Recall: {(rec*100):.1f}%  F1: {(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0):.3f}")
    print(f"False positives: {fp}  False negatives: {fn}")
    print("\nPer-pattern (tp/fp/tn/fn):")
    for p, c in sorted(per_pattern.items()):
        tot_att = c['tp']+c['fn']
        rate = (c['tp']/tot_att*100) if tot_att>0 else 0.0
        print(f"  - {p}: {c['tp']}/{tot_att} ({rate:.0f}%)  fp={c['fp']} fn={c['fn']}")


if __name__ == '__main__':
    main()

