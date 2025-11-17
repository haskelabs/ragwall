#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from typing import List

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


BASE = "https://api.fda.gov/drug/label.json"


def fetch_labels(drug: str, limit: int) -> List[dict]:
    if requests is None:
        raise SystemExit('Please: pip install requests (or run where network is allowed)')
    rows: List[dict] = []
    per_page = 100
    got = 0
    while got < limit:
        n = min(per_page, limit - got)
        params = {
            'search': f'openfda.brand_name:{drug}',
            'limit': str(n),
            'skip': str(got),
        }
        r = requests.get(BASE, params=params, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        results = data.get('results') or []
        if not results:
            break
        rows.extend(results)
        got += len(results)
        time.sleep(0.2)  # be gentle
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--categories', default='warfarin,metformin,lisinopril,atorvastatin,amoxicillin')
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(',') if c.strip()]
    all_docs = []
    for c in cats:
        docs = fetch_labels(c, args.limit)
        # Normalize
        for d in docs:
            did = d.get('id') or d.get('set_id') or d.get('openfda', {}).get('spl_id', [''])[0]
            text_parts = []
            for key in ['indications_and_usage', 'dosage_and_administration', 'contraindications', 'warnings', 'precautions']:
                val = d.get(key)
                if isinstance(val, list):
                    text_parts.append(f"{key.upper()}\n" + "\n".join(val))
            text = "\n\n".join(text_parts)
            all_docs.append({
                'id': f'openfda_{did}_{c}',
                'title': f'FDA Drug Label — {c}',
                'text': text or json.dumps(d),
                'labels': {'malicious': False, 'pattern': 'benign', 'domain': 'healthcare'}
            })

    with open(args.out, 'w') as f:
        for obj in all_docs:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'Wrote {len(all_docs)} FDA drug label docs to {args.out}')


if __name__ == '__main__':
    main()

