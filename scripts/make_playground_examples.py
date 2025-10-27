#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, random
from collections import defaultdict


def load_queries(path: str):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = str(obj.get('query', '')).strip()
            if not q:
                continue
            items.append({
                'query': q,
                'label': obj.get('label') or obj.get('type') or ('attacked' if obj.get('pattern') else 'benign'),
                'pattern': obj.get('pattern') or 'benign',
                'lang': obj.get('lang') or obj.get('language') or 'en',
            })
    return items


PATTERN_ORDER = [
    'benign',
    'role_play',
    'ignore',
    'no_rules',
    'developer_mode',
    'as_root',
    'secrets_printenv',
    'output_everything',
    'base64_blob',
    'system_prompt',
    'cot_exfil',
    'dan',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='data/queries.jsonl')
    ap.add_argument('--out', dest='out', default='public/samples/queries_top50.json')
    ap.add_argument('--total', type=int, default=50)
    ap.add_argument('--attacked', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    items = load_queries(args.inp)
    if not items:
        print(f'No queries found in {args.inp}', file=sys.stderr)
        sys.exit(1)

    # Partition
    attacked = [x for x in items if (x.get('label') or '').lower().startswith('attack') or (x.get('pattern') not in (None, '', 'benign'))]
    benign = [x for x in items if x not in attacked]

    # Group attacked by pattern
    by_pat = defaultdict(list)
    for x in attacked:
        by_pat[x.get('pattern') or 'other'].append(x)
    for v in by_pat.values():
        rnd.shuffle(v)

    # Desired counts
    n_total = max(1, int(args.total))
    n_att = min(len(attacked), max(0, int(args.attacked)))
    n_ben = min(len(benign), n_total - n_att)
    if n_att + n_ben < n_total:
        # fill remainder from whichever side has more
        extra = n_total - (n_att + n_ben)
        give = min(extra, len(attacked) - n_att)
        n_att += give
        n_ben += (extra - give)

    # Evenly sample attacked across patterns
    pats = [p for p in PATTERN_ORDER if p in by_pat] + [p for p in by_pat.keys() if p not in PATTERN_ORDER]
    sel_att: list[dict] = []
    if pats and n_att > 0:
        i = 0
        while len(sel_att) < n_att and any(by_pat[p] for p in pats):
            p = pats[i % len(pats)]
            if by_pat[p]:
                sel_att.append(by_pat[p].pop())
            i += 1

    # Sample benign
    rnd.shuffle(benign)
    sel_ben = benign[:n_ben]

    # Order: easy -> hard = benign first, then attacked by pattern order
    order_key = {name: idx for idx, name in enumerate(PATTERN_ORDER)}
    sel_att.sort(key=lambda x: order_key.get(x.get('pattern') or 'zzz', 999))
    out = sel_ben + sel_att

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(out)} examples to {args.out} (benign={len(sel_ben)}, attacked={len(sel_att)})')


if __name__ == '__main__':
    main()

