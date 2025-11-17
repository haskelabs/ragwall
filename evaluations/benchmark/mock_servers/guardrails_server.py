#!/usr/bin/env python3
"""
Mock Guardrails/llm-guard server - simulates guardrails-ai validation
Uses PII detection + basic prompt injection patterns
"""
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Guardrails-style validators
PII_PATTERNS = [
    r'\bssn\b',
    r'\bsocial\s+security',
    r'\bmedical\s+record\s+number\b',
    r'\bpatient\s+id\b',
]

SAFETY_PATTERNS = [
    r'ignore\s+',
    r'bypass\s+(hipaa|security|safety)',
    r'override\s+',
    r'lethal\s+dos',
    r'self-harm',
    r'without\s+prescription',
]

PII_COMPILED = [re.compile(p, re.IGNORECASE) for p in PII_PATTERNS]
SAFETY_COMPILED = [re.compile(p, re.IGNORECASE) for p in SAFETY_PATTERNS]


def validate_query(query: str) -> tuple[bool, str]:
    """Return (ok, reason)"""
    # Check for PII requests
    for pattern in PII_COMPILED:
        if pattern.search(query):
            return False, 'PII_DETECTED'

    # Check for safety violations
    for pattern in SAFETY_COMPILED:
        if pattern.search(query):
            return False, 'SAFETY_VIOLATION'

    return True, 'OK'


class GuardrailsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/sanitize':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            query = data.get('query', '')

            ok, reason = validate_query(query)

            response = {
                'ok': ok,
                'risky': not ok,
                'reason': reason,
                'clean_query': query if ok else ''
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    port = 9200
    server = HTTPServer(('127.0.0.1', port), GuardrailsHandler)
    print(f'Guardrails mock server running on port {port}')
    server.serve_forever()
