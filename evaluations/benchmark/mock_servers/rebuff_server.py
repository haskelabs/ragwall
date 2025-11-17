#!/usr/bin/env python3
"""
Mock Rebuff server - simulates Rebuff's prompt injection detection
Uses a simple keyword-based approach with scoring
"""
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Rebuff-style detection patterns (simplified)
INJECTION_PATTERNS = [
    r'ignore\s+(previous|all|above)',
    r'system\s+override',
    r'developer\s+mode',
    r'act\s+as',
    r'pretend\s+(you\s+are|to\s+be)',
    r'reveal\s+(system|hidden)',
    r'password|secret|credentials?',
    r'jailbreak',
    r'DAN\s+mode',
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(query: str) -> tuple[bool, float]:
    """Return (is_injection, score)"""
    hits = sum(1 for pattern in COMPILED_PATTERNS if pattern.search(query))
    score = min(hits / 3.0, 1.0)  # Normalize to 0-1
    risky = score >= 0.33  # Lower threshold than RAGWall
    return risky, score


class RebuffHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/detect':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            query = data.get('query', '')

            risky, score = detect_injection(query)

            response = {
                'risky': risky,
                'score': score,
                'threshold': 0.33
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs


if __name__ == '__main__':
    port = 9100
    server = HTTPServer(('127.0.0.1', port), RebuffHandler)
    print(f'Rebuff mock server running on port {port}')
    server.serve_forever()
