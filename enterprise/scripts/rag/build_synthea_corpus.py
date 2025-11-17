#!/usr/bin/env python3
import argparse, json, os, csv, random
from typing import List, Dict


def load_conditions(csv_path: str, limit: int = 500) -> List[Dict]:
    docs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            desc = row.get('DESCRIPTION', '')
            code = row.get('CODE', '')
            start = row.get('START', '')
            stop = row.get('STOP', 'ongoing')

            text = f"Condition: {desc}\nSNOMED Code: {code}\nOnset: {start}\nStatus: {'Resolved' if stop else 'Active'}"
            docs.append({
                'id': f'synthea_cond_{i:05d}',
                'text': text,
                'category': 'clinical_condition',
                'labels': {'malicious': False}
            })
    return docs


def load_medications(csv_path: str, limit: int = 500) -> List[Dict]:
    docs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            desc = row.get('DESCRIPTION', '')
            code = row.get('CODE', '')

            text = f"Medication: {desc}\nRxNorm Code: {code}\nPrescription guidelines apply. Check for contraindications and drug interactions before administering."
            docs.append({
                'id': f'synthea_med_{i:05d}',
                'text': text,
                'category': 'medication',
                'labels': {'malicious': False}
            })
    return docs


def load_procedures(csv_path: str, limit: int = 300) -> List[Dict]:
    docs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            desc = row.get('DESCRIPTION', '')
            code = row.get('CODE', '')

            text = f"Procedure: {desc}\nSNOMED Code: {code}\nFollow standard clinical protocols and obtain informed consent."
            docs.append({
                'id': f'synthea_proc_{i:05d}',
                'text': text,
                'category': 'procedure',
                'labels': {'malicious': False}
            })
    return docs


def main():
    ap = argparse.ArgumentParser(description="Build healthcare corpus from Synthea CSV files")
    ap.add_argument('--synthea-dir', required=True, help='Path to Synthea CSV directory')
    ap.add_argument('--out', required=True, help='Output JSONL corpus file')
    ap.add_argument('--conditions-limit', type=int, default=500)
    ap.add_argument('--medications-limit', type=int, default=500)
    ap.add_argument('--procedures-limit', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    conditions_csv = os.path.join(args.synthea_dir, 'conditions.csv')
    medications_csv = os.path.join(args.synthea_dir, 'medications.csv')
    procedures_csv = os.path.join(args.synthea_dir, 'procedures.csv')

    print(f"Loading conditions from {conditions_csv}...")
    conditions = load_conditions(conditions_csv, args.conditions_limit)

    print(f"Loading medications from {medications_csv}...")
    medications = load_medications(medications_csv, args.medications_limit)

    print(f"Loading procedures from {procedures_csv}...")
    procedures = load_procedures(procedures_csv, args.procedures_limit)

    all_docs = conditions + medications + procedures
    random.shuffle(all_docs)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(all_docs)} documents to {args.out}")
    print(f"  - Conditions: {len(conditions)}")
    print(f"  - Medications: {len(medications)}")
    print(f"  - Procedures: {len(procedures)}")


if __name__ == '__main__':
    main()