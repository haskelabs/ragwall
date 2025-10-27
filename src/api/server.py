#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import re
import sys
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

# Make sure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sanitizer.rag_sanitizer import QuerySanitizer, SanitizerConfig  # noqa: E402
from sanitizer.jailbreak.prr_gate import DEFAULT_KEYWORDS, DEFAULT_STRUCTURE  # noqa: E402
import hmac
import base64
import hashlib as _hashlib


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _stable_hash(s: str) -> int:
    return int(_hashlib.sha256(s.encode('utf-8')).hexdigest()[:8], 16)


def _hmac_tag(val: str, key: str, kind: str) -> str:
    mac = hmac.new(key.encode('utf-8'), val.encode('utf-8'), _hashlib.sha256).digest()
    b32 = base64.b32encode(mac)[:10].decode('ascii').rstrip('=')
    return f"<{kind}:{b32}>"


EMAIL_RX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
TOKEN_RX = re.compile(r"sk-[A-Za-z0-9]{20,}")
SECRET_RX = re.compile(r"(?i)(api|auth|secret|token|key)[=:]\s*([A-Za-z0-9_\-]{12,})")


class RagWallService:
    def __init__(self,
                 vectors_path: str | None = None,
                 model_name: str = 'distilgpt2',
                 layer: str = 'transformer.h.1',
                 mode: str = 'auto') -> None:
        # Mode: 'rules_only' (regex/structure only), 'full' (use model), 'auto' (two-stage)
        self.mode = (mode or 'auto').lower()
        # HIPAA mode toggles privacy defaults and receipts
        self.hipaa_mode = os.environ.get('RAGWALL_HIPAA_MODE', '0').strip() in ('1', 'true', 'True')
        # Full sanitizer (model-enabled)
        cfg_full = SanitizerConfig(
            model_name=model_name,
            vectors_path=vectors_path or '',
            layer=layer,
            pool_k=6,
            max_edit_positions=6,
            scale=0.2,
            pattern_gate=True,
            no_overall=True,
            orthogonalize=True,
            entropy_scale=False,
            quorum=2,
            healthcare_mode=self.hipaa_mode or (os.environ.get('RAGWALL_HEALTHCARE','0').strip() in ('1','true','True')),
        )
        self.sanitizer_full = QuerySanitizer(cfg_full)
        # Rules-only sanitizer (force no model by passing invalid model name)
        cfg_rules = SanitizerConfig(
            model_name='__disabled__',
            vectors_path='',
            layer=layer,
            pool_k=6,
            max_edit_positions=6,
            scale=0.2,
            pattern_gate=True,
            no_overall=True,
            orthogonalize=True,
            entropy_scale=False,
            quorum=2,
            healthcare_mode=self.hipaa_mode or (os.environ.get('RAGWALL_HEALTHCARE','0').strip() in ('1','true','True')),
        )
        self.sanitizer_rules = QuerySanitizer(cfg_rules)
        # Default risk lexicon for document-side cues
        self.risk_lexicon: List[re.Pattern] = [
            re.compile(r"(?i)\bignore\s+(previous|prior|all|any)\s+(instructions?|rules?|policies)", re.IGNORECASE),
            re.compile(r"(?i)\bdisregard\b", re.IGNORECASE),
            re.compile(r"(?i)\bno\s+rules?\b", re.IGNORECASE),
            re.compile(r"(?i)\bwithout\s+restrictions\b", re.IGNORECASE),
            re.compile(r"(?i)\bdeveloper\s+mode\b", re.IGNORECASE),
            re.compile(r"(?i)\byou\s+are\s+dan\b|\bDAN\b", re.IGNORECASE),
            re.compile(r"(?i)\bact\s+as\s+(root|admin|sudo)", re.IGNORECASE),
            re.compile(r"(?i)\b(printenv|dump\s+all|reveal\s+all)\b", re.IGNORECASE),
            re.compile(r"(?i)\bbase64\b", re.IGNORECASE),
            re.compile(r"(?i)IGNORE\s+PREVIOUS\s+INSTRUCTIONS", re.IGNORECASE),
            # Healthcare-specific risky directives
            re.compile(r"(?i)\boverride\s+(dosage|dose|protocols?|contraindications?)\b"),
            re.compile(r"(?i)\bignore\s+(allerg(y|ies)|contraindications?)\b"),
            re.compile(r"(?i)\bbypass\s+(consent|hipaa|privacy)\b"),
            re.compile(r"(?i)\breveal\s+(patient|phi|identifiers?)\b"),
            re.compile(r"(?i)\bexport\s+(patient|record|ehr|emr|phi)\b"),
            re.compile(r"(?i)\bde\-?identify\s*[:=]?\s*false\b"),
        ]

    def _rules_hit(self, text: str) -> bool:
        # Fast check: keyword/structure patterns only
        for pat in DEFAULT_KEYWORDS + DEFAULT_STRUCTURE:
            try:
                if re.search(pat, text, re.I):
                    return True
            except Exception:
                continue
        return False

    # PHI/PII detectors for Privacy Mode (HMAC pseudonymization)
    SSN_RX = re.compile(r"\b(\d{3}[- ]?\d{2}[- ]?\d{4})\b")
    PHONE_RX = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    DOB_RX = re.compile(r"\b(?:(?:\d{4}[-/.]\d{2}[-/.]\d{2})|(?:\d{2}[-/.]\d{2}[-/.]\d{4}))\b")
    MRN_RX = re.compile(r"(?i)\b(MRN|Medical\s*Record\s*Number)[:#]?\s*([A-Z0-9\-]{6,})\b")
    NPI_RX = re.compile(r"(?i)\b(NPI)[:#]?\s*(\d{10})\b")
    CLAIM_RX = re.compile(r"(?i)\b(Claim\s*ID|Authorization\s*ID|Case\s*Number)[:#]?\s*([A-Z0-9\-]{6,})\b")
    ADDRESS_RX = re.compile(r"(?i)\b(\d{2,5}\s+[A-Za-z0-9\.\-\s]+\b(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Ln\.|Drive|Dr\.|Court|Ct\.|Way))\b")
    PATIENT_NAME_RX = re.compile(r"(?i)\b(Patient|Member|Subscriber)[:#]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

    def sanitize(self, query: str, privacy: bool = False, hmac_key: str | None = None) -> Dict[str, Any]:
        # Choose backend per mode
        if self.mode == 'rules_only':
            sanitized, meta = self.sanitizer_rules.sanitize_query(query)
        elif self.mode == 'full':
            sanitized, meta = self.sanitizer_full.sanitize_query(query)
        else:
            # auto: run rules first; only invoke model if rules fire
            if self._rules_hit(query):
                sanitized, meta = self.sanitizer_full.sanitize_query(query)
            else:
                sanitized, meta = self.sanitizer_rules.sanitize_query(query)
        # Patterns: use the sanitizer's internal pattern selection for informative tags
        try:
            patterns: List[str] = self.sanitizer_rules._select_patterns(query)  # type: ignore[attr-defined]
        except Exception:
            patterns = []
        reasons: List[str] = []
        for p in patterns[:5]:
            reasons.append(f"lexicon:{p}")

        sanitized_equal = (sanitized == query)

        # Optional Privacy Mode (span-level pseudonymization)
        if privacy or self.hipaa_mode:
            key = hmac_key or os.environ.get('RAGWALL_HMAC_KEY', 'ragwall-default')
            def _sub_email(m: re.Match[str]) -> str:
                reasons.append('pii:email')
                return _hmac_tag(m.group(0), key, 'email')
            def _sub_token(m: re.Match[str]) -> str:
                reasons.append('secret:token')
                return _hmac_tag(m.group(0), key, 'token')
            def _sub_secret(m: re.Match[str]) -> str:
                reasons.append('secret:key')
                prefix = m.group(1)
                return f"{prefix}:" + _hmac_tag(m.group(2), key, 'key')
            def _sub_ssn(m: re.Match[str]) -> str:
                reasons.append('phi:ssn')
                return _hmac_tag(m.group(1), key, 'ssn')
            def _sub_phone(m: re.Match[str]) -> str:
                reasons.append('phi:phone')
                return _hmac_tag(m.group(0), key, 'phone')
            def _sub_dob(m: re.Match[str]) -> str:
                reasons.append('phi:dob')
                return _hmac_tag(m.group(0), key, 'dob')
            def _sub_mrn(m: re.Match[str]) -> str:
                reasons.append('phi:mrn')
                return m.group(1) + ':' + _hmac_tag(m.group(2), key, 'mrn')
            def _sub_npi(m: re.Match[str]) -> str:
                reasons.append('phi:npi')
                return m.group(1) + ':' + _hmac_tag(m.group(2), key, 'npi')
            def _sub_claim(m: re.Match[str]) -> str:
                reasons.append('phi:claim')
                return m.group(1) + ':' + _hmac_tag(m.group(2), key, 'claim')
            def _sub_address(m: re.Match[str]) -> str:
                reasons.append('phi:address')
                return _hmac_tag(m.group(1), key, 'address')
            def _sub_patient_name(m: re.Match[str]) -> str:
                reasons.append('phi:name')
                return m.group(1) + ': ' + _hmac_tag(m.group(2), key, 'name')
            sanitized = EMAIL_RX.sub(_sub_email, sanitized)
            sanitized = TOKEN_RX.sub(_sub_token, sanitized)
            sanitized = SECRET_RX.sub(_sub_secret, sanitized)
            sanitized = self.SSN_RX.sub(_sub_ssn, sanitized)
            sanitized = self.PHONE_RX.sub(_sub_phone, sanitized)
            sanitized = self.DOB_RX.sub(_sub_dob, sanitized)
            sanitized = self.MRN_RX.sub(_sub_mrn, sanitized)
            sanitized = self.NPI_RX.sub(_sub_npi, sanitized)
            sanitized = self.CLAIM_RX.sub(_sub_claim, sanitized)
            sanitized = self.ADDRESS_RX.sub(_sub_address, sanitized)
            sanitized = self.PATIENT_NAME_RX.sub(_sub_patient_name, sanitized)
        # Minimal audit receipt (no raw text)
        import time
        receipt = {
            'ts': int(time.time()),
            'hipaa_mode': self.hipaa_mode,
            'risky': bool(meta.get('risky', False)),
            'families_hit': list(meta.get('families_hit', [])) if isinstance(meta.get('families_hit', []), list) else [],
            'patterns': patterns[:5],
            'hashes': {
                'original_sha256': sha256_hex(query),
                'sanitized_sha256': sha256_hex(sanitized),
            },
            'config': {
                'mode': self.mode,
                'model': getattr(self.sanitizer_full.cfg, 'model_name', ''),
                'layer': getattr(self.sanitizer_full.cfg, 'layer', ''),
            }
        }

        return {
            'sanitized_for_embed': sanitized,
            'risky': bool(meta.get('risky', False)),
            'patterns': patterns,
            'hashes': {
                'original_sha256': sha256_hex(query),
                'sanitized_sha256': sha256_hex(sanitized),
            },
            'meta': {**meta, 'sanitized_equal': sanitized_equal, 'reasons': reasons},
            'receipt': receipt if (os.environ.get('RAGWALL_AUDIT_RECEIPTS', '0') in ('1','true','True') or self.hipaa_mode) else None,
        }

    def _doc_looks_risky(self, text: str) -> bool:
        for rx in self.risk_lexicon:
            if rx.search(text):
                return True
        # Heuristic base64 block: long continuous base64-ish run
        if re.search(r"(?:[A-Za-z0-9+/]{24,}=*)", text):
            return True
        return False

    def rerank(self,
               risky: bool,
               baseline_hrcr_positive: bool,
               k: int,
               penalty: float,
               candidates: List[Dict[str, Any]],
               fallback: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        # Build groups with deterministic tie-breaking epsilon
        items: List[Dict[str, Any]] = []
        penalized_ids: List[str] = []
        safe: List[Dict[str, Any]] = []
        risky_docs: List[Dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            cid = str(c.get('id')) if c.get('id') is not None else None
            blob = ' '.join([str(c.get('title', '')), str(c.get('snippet', '')), str(c.get('text', ''))])
            looks_risky = self._doc_looks_risky(blob)
            base_score = float(c.get('score', 0.0))
            eps = 1e-9 * _stable_hash(cid or str(idx))
            c['_stable_score'] = base_score + eps
            c['_reasons'] = ['doc:risky-lexicon'] if looks_risky else []
            if looks_risky:
                risky_docs.append(c)
                if cid is not None:
                    penalized_ids.append(cid)
            else:
                safe.append(c)
        # No mask: return original order deterministically
        if not (risky and baseline_hrcr_positive):
            ids_sorted = [c.get('id') for c in candidates][: max(0, int(k)) or len(candidates)]
            items = [
                {'id': c.get('id'), 'demoted': False, 'reasons': c.get('_reasons', []), 'score': c.get('_stable_score', 0.0)}
                for c in candidates[:len(ids_sorted)]
            ]
            return {'ids_sorted': ids_sorted, 'penalized': [], 'items': items, 'safe_topk': ids_sorted}
        # Risk-aware grouping
        new_order = safe + risky_docs
        ids_sorted = [c.get('id') for c in new_order][: max(0, int(k)) or len(new_order)]
        # Fallback fill if needed
        if len(ids_sorted) < k and fallback:
            have = set(str(x) for x in ids_sorted)
            for c in fallback:
                cid = str(c.get('id')) if c.get('id') is not None else None
                if cid and cid not in have:
                    ids_sorted.append(cid)
                    have.add(cid)
                    if len(ids_sorted) >= k:
                        break
        # Items
        items = []
        for c in new_order:
            items.append({'id': c.get('id'), 'demoted': str(c.get('id')) in penalized_ids, 'reasons': c.get('_reasons', []), 'score': c.get('_stable_score', 0.0)})
        out = {'ids_sorted': ids_sorted, 'penalized': penalized_ids, 'items': items, 'safe_topk': [c.get('id') for c in safe][:k]}
        # Optional audit receipt for rerank
        if os.environ.get('RAGWALL_AUDIT_RECEIPTS', '0') in ('1','true','True') or self.hipaa_mode:
            import time
            out['receipt'] = {
                'ts': int(time.time()),
                'hipaa_mode': self.hipaa_mode,
                'risky': bool(risky),
                'baseline_hrcr_positive': bool(baseline_hrcr_positive),
                'penalized_sha256': [sha256_hex(str(x)) for x in penalized_ids],
                'config': {'mode': self.mode}
            }
        return out


def _json_error(handler: BaseHTTPRequestHandler, code: int, msg: str) -> None:
    payload = json.dumps({'error': msg}).encode('utf-8')
    handler.send_response(code)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.send_header('Content-Length', str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class RagWallHandler(BaseHTTPRequestHandler):
    service: RagWallService | None = None

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
        sys.stderr.write("[ragwall] " + fmt % args + "\n")

    def _read_json(self) -> Tuple[Dict[str, Any], str]:
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b''
        body = raw.decode('utf-8')
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        return data, body

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == '/health' or self.path == '/v1/health':
            payload = json.dumps({'status': 'ok'}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        _json_error(self, 404, 'Not Found')

    def do_POST(self) -> None:  # noqa: N802
        svc = RagWallHandler.service
        if svc is None:
            _json_error(self, 500, 'Service not initialized')
            return
        data, _ = self._read_json()
        if self.path == '/v1/sanitize':
            q = data.get('query')
            if not isinstance(q, str) or not q.strip():
                _json_error(self, 400, 'Field "query" must be a non-empty string')
                return
            privacy = bool(data.get('privacy', False))
            hmac_key = data.get('hmac_key') if isinstance(data.get('hmac_key'), str) else None
            out = svc.sanitize(q, privacy=privacy, hmac_key=hmac_key)
            payload = json.dumps(out).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == '/v1/rerank':
            risky = bool(data.get('risky', False))
            baseline_hrcr_positive = bool(data.get('baseline_hrcr_positive', False))
            k = int(data.get('k', 5))
            penalty = float(data.get('penalty', 0.2))
            candidates = data.get('candidates')
            if not isinstance(candidates, list):
                _json_error(self, 400, 'Field "candidates" must be a list')
                return
            fallback = data.get('fallback') if isinstance(data.get('fallback'), list) else None
            out = svc.rerank(risky=risky,
                             baseline_hrcr_positive=baseline_hrcr_positive,
                             k=k,
                             penalty=penalty,
                             candidates=candidates,
                             fallback=fallback)
            payload = json.dumps(out).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        _json_error(self, 404, 'Not Found')


def run(host: str = '127.0.0.1', port: int = 8000) -> None:
    vectors = os.environ.get('RAGWALL_VECTORS', '').strip()
    if not vectors:
        # Default to tiny demo vectors if present
        default_vec = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'tiny_jb_vectors.pt')
        if os.path.exists(default_vec):
            vectors = default_vec
    model_name = os.environ.get('RAGWALL_MODEL', 'distilgpt2').strip() or 'distilgpt2'
    mode = os.environ.get('RAGWALL_MODE', 'auto').strip() or 'auto'
    svc = RagWallService(vectors_path=vectors, model_name=model_name, mode=mode)
    RagWallHandler.service = svc
    server = HTTPServer((host, port), RagWallHandler)
    print(f"[ragwall] Listening on http://{host}:{port} (mode={mode}, model={model_name}, vectors={vectors or 'none'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    run()
