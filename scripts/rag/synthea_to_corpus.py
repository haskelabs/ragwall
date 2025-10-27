#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os
from typing import Dict, List, Any, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        return [dict(r) for r in rdr]


def summarize_patient(pid: str,
                      patient_row: Dict[str, str] | None,
                      conditions: List[Dict[str, str]],
                      medications: List[Dict[str, str]],
                      procedures: List[Dict[str, str]],
                      allergies: List[Dict[str, str]]) -> Tuple[str, str]:
    title = f"Synthetic Patient Record — {pid}"
    lines: List[str] = []
    # High-level demographics (no SSN, no exact address)
    if patient_row:
        first = patient_row.get('FIRST', '')
        last = patient_row.get('LAST', '')
        gender = patient_row.get('GENDER', '')
        birthyear = (patient_row.get('BIRTHDATE', '') or '')[:4]
        city = patient_row.get('CITY', '')
        state = patient_row.get('STATE', '')
        lines += [
            f"Patient: {first} {last} (synthetic)",
            f"Gender: {gender}  Birth Year: {birthyear}",
            f"Location: {city}, {state}",
        ]
    # Conditions
    if conditions:
        lines.append("\nConditions:")
        for c in conditions[:50]:
            onset = c.get('START') or c.get('DATE') or ''
            code = c.get('CODE', '')
            desc = c.get('DESCRIPTION', '')
            lines.append(f"- {desc} (code {code}) since {onset}")
    # Medications
    if medications:
        lines.append("\nMedications:")
        for m in medications[:50]:
            start = m.get('START', '')
            stop = m.get('STOP', '')
            code = m.get('CODE', '')
            desc = m.get('DESCRIPTION', '')
            lines.append(f"- {desc} (code {code}) from {start} to {stop or 'present'}")
    # Procedures
    if procedures:
        lines.append("\nProcedures:")
        for p in procedures[:50]:
            date = p.get('DATE', '')
            code = p.get('CODE', '')
            desc = p.get('DESCRIPTION', '')
            lines.append(f"- {desc} (code {code}) on {date}")
    # Allergies
    if allergies:
        lines.append("\nAllergies:")
        for a in allergies[:50]:
            cause = a.get('DESCRIPTION', '')
            lines.append(f"- {cause}")
    text = "\n".join(lines)
    return title, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True, help='Path to Synthea CSV directory (e.g., data/healthcare/csv)')
    ap.add_argument('--out', required=True, help='Output JSONL corpus path')
    ap.add_argument('--limit', type=int, default=0, help='Optional max patients to export')
    args = ap.parse_args()

    base = args.in_dir
    patients = {r['Id']: r for r in read_csv(os.path.join(base, 'patients.csv'))}
    conds = read_csv(os.path.join(base, 'conditions.csv')) if os.path.exists(os.path.join(base, 'conditions.csv')) else []
    meds = read_csv(os.path.join(base, 'medications.csv')) if os.path.exists(os.path.join(base, 'medications.csv')) else []
    procs = read_csv(os.path.join(base, 'procedures.csv')) if os.path.exists(os.path.join(base, 'procedures.csv')) else []
    allg = read_csv(os.path.join(base, 'allergies.csv')) if os.path.exists(os.path.join(base, 'allergies.csv')) else []

    by_patient_conds: Dict[str, List[Dict[str, str]]] = {}
    for c in conds:
        by_patient_conds.setdefault(c.get('PATIENT',''), []).append(c)
    by_patient_meds: Dict[str, List[Dict[str, str]]] = {}
    for m in meds:
        by_patient_meds.setdefault(m.get('PATIENT',''), []).append(m)
    by_patient_procs: Dict[str, List[Dict[str, str]]] = {}
    for p in procs:
        by_patient_procs.setdefault(p.get('PATIENT',''), []).append(p)
    by_patient_allg: Dict[str, List[Dict[str, str]]] = {}
    for a in allg:
        by_patient_allg.setdefault(a.get('PATIENT',''), []).append(a)

    ids = list(patients.keys())
    if args.limit and args.limit > 0:
        ids = ids[: int(args.limit)]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    n = 0
    with open(args.out, 'w') as fout:
        for pid in ids:
            pr = patients.get(pid)
            cs = by_patient_conds.get(pid, [])
            ms = by_patient_meds.get(pid, [])
            ps = by_patient_procs.get(pid, [])
            as_ = by_patient_allg.get(pid, [])
            title, text = summarize_patient(pid, pr, cs, ms, ps, as_)
            doc = {
                'id': f'pt_{pid}',
                'title': title,
                'text': text,
                'labels': {'malicious': False, 'pattern': 'benign', 'domain': 'healthcare'}
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
            n += 1
    print(f'Wrote {n} patient docs to {args.out}')


if __name__ == '__main__':
    main()

