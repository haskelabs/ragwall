#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path
from string import Template

TEMPLATE = Template("""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>RagWall Evaluation Report</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    :root{ --ink:#0f172a; --muted:#64748b; --line:#e5e7eb; --bg:#ffffff; --accent:#4F46E5; --good:#10B981; }
    @media(prefers-color-scheme:dark){ :root{ --ink:#e5e7eb; --muted:#94a3b8; --line:#1f2937; --bg:#0b1220; } }
    body{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; color:var(--ink); background:var(--bg); margin:0; }
    .wrap{ max-width: 960px; margin: 0 auto; padding: 24px; }
    h1{ font-size: 28px; margin: 8px 0 4px; }
    .muted{ color: var(--muted); }
    .card{ border:1px solid var(--line); border-radius: 12px; padding: 16px; margin: 16px 0; }
    .grid{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    table{ width:100%; border-collapse: collapse; }
    th, td{ text-align:left; padding:8px; border-bottom:1px solid var(--line); }
    .bar{ height: 10px; background: #e5e7eb; border-radius: 6px; position: relative; overflow:hidden; }
    .bar > span{ position:absolute; top:0; left:0; height:100%; background: var(--accent); }
    .bar-good > span{ background: var(--good); }
    .pill{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:4px 8px; font-size:12px; margin-right:6px; }
    .small{ font-size: 12px; }
    .title{ display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .hr{ height:1px; background:var(--line); margin:16px 0; }
    .foot{ margin-top: 24px; font-size:12px; color:var(--muted); }
    @media print{ .wrap{ padding:0; } .card{ break-inside: avoid; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class=\"title\">
      <h1>RagWall Evaluation Report</h1>
      <div class=\"muted small\">Generated from $summary_path</div>
    </div>
    <div class=\"muted\">A/B comparison of baseline vs. sanitized+rerank retrieval.</div>

    <div class=\"grid\">
      <div class=\"card\">
        <h3>HRCR@5</h3>
        <div class=\"small\">Baseline: $hr5_b &nbsp;&nbsp; Sanitized: $hr5_s &nbsp;&nbsp; Δrel ≈ $hr5_rel</div>
        <div class=\"bar\" title=\"baseline\"><span style=\"width:$hr5_b_pct%\"></span></div>
        <div class=\"bar bar-good\" title=\"sanitized\" style=\"margin-top:6px\"><span style=\"width:$hr5_s_pct%\"></span></div>
        <div class=\"small muted\" style=\"margin-top:6px\">CI (rel drop) ≈ $hr5_ci_lo – $hr5_ci_hi</div>
      </div>
      <div class=\"card\">
        <h3>HRCR@10</h3>
        <div class=\"small\">Baseline: $hr10_b &nbsp;&nbsp; Sanitized: $hr10_s &nbsp;&nbsp; Δrel ≈ $hr10_rel</div>
        <div class=\"bar\" title=\"baseline\"><span style=\"width:$hr10_b_pct%\"></span></div>
        <div class=\"bar bar-good\" title=\"sanitized\" style=\"margin-top:6px\"><span style=\"width:$hr10_s_pct%\"></span></div>
      </div>
    </div>

    <div class=\"card\">
      <h3>Benign Drift</h3>
      <div>Jaccard@5 (benign): <span class=\"pill\">mean $j5_mean</span> <span class=\"pill\">drift $j5_drift</span></div>
    </div>

    <div class=\"card\">
      <h3>Configuration</h3>
      <table>
        <tr><th>Embedder</th><td>$embedder / $embedder_model</td></tr>
        <tr><th>Penalty</th><td>$penalty</td></tr>
        <tr><th>Mask Condition</th><td>$mask</td></tr>
        <tr><th>Bootstrap Samples</th><td>$bs</td></tr>
        <tr><th>Seed</th><td>$seed</td></tr>
      </table>
    </div>

    <div class=\"card\">
      <h3>Per‑Pattern HRCR (baseline → sanitized)</h3>
      <table>
        <thead><tr><th>Pattern</th><th>Count</th><th>Baseline</th><th>Sanitized</th><th>Δrel</th></tr></thead>
        <tbody>
         $rows
        </tbody>
      </table>
      <div class=\"small muted\" style=\"margin-top:6px\">Small counts have wide uncertainty; interpret cautiously.</div>
    </div>

    <div class=\"foot\">
      Notes: HRCR@k is the fraction of risky docs in top‑k. Relative drops and CIs come from bootstrap samples when available. Results vary by corpus, embedder, and configuration. Print this page to PDF for sharing.
    </div>
  </div>
</body>
</html>
""")

def main():
    if len(sys.argv) < 3:
        print("Usage: make_report_html.py <summary_json> <out_html>")
        sys.exit(2)
    summary_path = Path(sys.argv[1])
    out_html = Path(sys.argv[2])
    data = json.loads(summary_path.read_text())

    hr5_b = float(data["HRCR@5"]["baseline"]) if "HRCR@5" in data else 0.0
    hr5_s = float(data["HRCR@5"]["sanitized"]) if "HRCR@5" in data else 0.0
    hr5_rel = float(data["HRCR@5"].get("relative_drop", 0.0))
    hr10_b = float(data["HRCR@10"]["baseline"]) if "HRCR@10" in data else 0.0
    hr10_s = float(data["HRCR@10"]["sanitized"]) if "HRCR@10" in data else 0.0
    hr10_rel = float(data["HRCR@10"].get("relative_drop", 0.0))

    j5_mean = float(data.get("Benign_Jaccard@5", {}).get("mean", 0.0))
    j5_drift = float(data.get("Benign_Jaccard@5", {}).get("drift", 0.0))

    bs = int(data.get("knobs", {}).get("bootstrap_samples", 0))
    seed = data.get("knobs", {}).get("seed", "")
    embedder = data.get("knobs", {}).get("embedder_backend", "")
    embedder_model = data.get("knobs", {}).get("embedder_model", "")
    penalty = data.get("knobs", {}).get("risk_rerank_penalty", "")
    mask = data.get("knobs", {}).get("mask_condition", "")

    # Bootstrap CI for HRCR@5 rel drop if present
    hr5_ci_lo = hr5_ci_hi = hr5_rel
    try:
        ci = data.get("bootstrap", {}).get("HRCR@5", {}).get("relative_drop_CI", [])
        if len(ci) == 3:
            hr5_ci_lo, _, hr5_ci_hi = float(ci[0]), float(ci[1]), float(ci[2])
    except Exception:
        pass

    rows = []
    per = data.get("per_pattern_HRCR", {})
    counts = data.get("per_pattern_counts", {})
    for pat, vals in per.items():
        b = float(vals.get("baseline", 0.0))
        s = float(vals.get("sanitized", 0.0))
        rel = float(vals.get("relative_drop", 0.0))
        c = int(counts.get(pat, 0))
        rows.append(f"<tr><td>{pat}</td><td>{c}</td><td>{b:.2f}</td><td>{s:.2f}</td><td>{rel:.0%}</td></tr>")

    html = TEMPLATE.substitute(
        summary_path=summary_path,
        hr5_b=hr5_b, hr5_s=hr5_s, hr5_rel=hr5_rel,
        hr10_b=hr10_b, hr10_s=hr10_s, hr10_rel=hr10_rel,
        hr5_b_pct=round(hr5_b*100, 1), hr5_s_pct=round(hr5_s*100, 1),
        hr10_b_pct=round(hr10_b*100, 1), hr10_s_pct=round(hr10_s*100, 1),
        hr5_ci_lo=hr5_ci_lo, hr5_ci_hi=hr5_ci_hi,
        j5_mean=j5_mean, j5_drift=j5_drift,
        embedder=embedder, embedder_model=embedder_model, penalty=penalty, mask=mask, bs=bs, seed=seed,
        rows="\n".join(rows)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding='utf-8')
    print(f"Wrote {out_html}")

if __name__ == '__main__':
    main()
