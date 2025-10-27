#!/usr/bin/env python3
from __future__ import annotations
import argparse, json


def stream_jsonl(path: str):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('inputs', nargs='+', help='One or more JSONL files to merge')
    args = ap.parse_args()

    seen = set()
    n = 0
    with open(args.out, 'w') as fout:
        for p in args.inputs:
            for obj in stream_jsonl(p):
                did = str(obj.get('id', ''))
                key = did or json.dumps(obj, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
                n += 1
    print(f'Merged {n} docs into {args.out}')


if __name__ == '__main__':
    main()

