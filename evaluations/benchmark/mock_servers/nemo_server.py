#!/usr/bin/env python3
"""
Mock NeMo Guardrails server - simulates NeMo's rule-based flow control
Uses structured rules with topic detection
"""
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# NeMo-style rules (topic + action patterns)
BLOCKED_ACTIONS = [
    r'(ignore|bypass|override|disable)\s+(instruction|policy|rule|safety|hipaa)',
    r'developer\s+mode',
    r'reveal\s+(password|secret|ssn|credentials)',
    r'(you\s+are\s+now|act\s+as|pretend)\s+.{0,20}(admin|root|DAN)',
]

SENSITIVE_TOPICS = [
    r'ssn',
    r'social\s+security\s+number',
    r'lethal\s+dos',
    r'self-harm',
    r'without\s+prescription',
]

ACTION_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_ACTIONS]
TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_TOPICS]


def check_guardrails(query: str) -> tuple[bool, str]:
    """Return (blocked, reason)"""
    # Check blocked actions
    for pattern in ACTION_COMPILED:
        if pattern.search(query):
            return True, 'BLOCKED_ACTION'

    # Check sensitive topics
    for pattern in TOPIC_COMPILED:
        if pattern.search(query):
            return True, 'SENSITIVE_TOPIC'

    return False, 'ALLOWED'


class NemoHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/detect':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            query = data.get('query', '')

            blocked, reason = check_guardrails(query)

            response = {
                'blocked': blocked,
                'risky': blocked,
                'reason': reason
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
    port = 9000
    server = HTTPServer(('127.0.0.1', port), NemoHandler)
    print(f'NeMo Guardrails mock server running on port {port}')
    server.serve_forever()
