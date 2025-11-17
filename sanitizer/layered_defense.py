#!/usr/bin/env python3
"""
Layered Defense System for RAGWall
===================================

Integrates multiple defense layers for comprehensive protection:
1. Content Safety Pre-filter (harmful content)
2. Pattern Detection (regex-based)
3. Semantic Analysis (intent understanding)
4. Context Tracking (multi-turn attacks)
5. Transformer Fallback (ML-based)
6. Rate Limiting & Monitoring

This provides defense-in-depth against various attack vectors.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from sanitizer.jailbreak.prr_gate import PRRGate
from sanitizer.rag_sanitizer import QuerySanitizer

# Import optional components with fallback
try:
    from sanitizer.ml.semantic_analyzer import SemanticAnalyzer, SemanticResult
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    SemanticAnalyzer = None
    SemanticResult = None

try:
    from sanitizer.context.conversation_tracker import ConversationTracker, ContextAnalysis
    HAS_CONTEXT = True
except ImportError:
    HAS_CONTEXT = False
    ConversationTracker = None
    ContextAnalysis = None

logger = logging.getLogger(__name__)


@dataclass
class LayeredDefenseResult:
    """Complete result from layered defense analysis."""

    # Core results
    is_attack: bool
    confidence: float
    sanitized_query: str

    # Layer-specific results
    content_safety_blocked: bool = False
    pattern_detected: bool = False
    semantic_attack: bool = False
    context_attack: bool = False
    ml_detected: bool = False

    # Detailed findings
    attack_types: List[str] = field(default_factory=list)
    blocked_by_layers: List[str] = field(default_factory=list)
    risk_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    # Monitoring data
    processing_time_ms: float = 0.0
    session_id: str = ""


class RateLimiter:
    """Simple rate limiter for abuse prevention."""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)

    def check_rate(self, identifier: str) -> Tuple[bool, str]:
        """Check if request is within rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60

        # Clean old requests
        self.requests[identifier] = [
            t for t in self.requests[identifier] if t > minute_ago
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False, f"Rate limit exceeded: {self.max_requests} requests per minute"

        # Add current request
        self.requests[identifier].append(current_time)
        return True, ""


class ContentSafetyFilter:
    """
    Pre-filter for harmful content requests.
    This addresses the JailbreakBench gap - detecting requests for harmful content.
    """

    def __init__(self):
        self.harmful_categories = {
            "violence": [
                "bomb", "weapon", "explosive", "kill", "murder", "assault",
                "terrorist", "violence", "harm", "hurt", "attack"
            ],
            "illegal": [
                "drug", "meth", "cocaine", "heroin", "illegal", "counterfeit",
                "fraud", "launder", "hack", "crack", "pirate"
            ],
            "hate": [
                "hate", "racist", "discrimination", "supremacy", "nazi",
                "extremist", "bigot", "slur"
            ],
            "self_harm": [
                "suicide", "self-harm", "cut myself", "end my life", "kill myself"
            ],
            "sexual": [
                "sexual", "porn", "explicit", "nude", "nsfw"
            ],
            "malware": [
                "ransomware", "virus", "trojan", "malware", "keylogger",
                "botnet", "ddos", "exploit"
            ]
        }

        self.harmful_phrases = [
            "how to make",
            "tutorial on",
            "instructions for",
            "guide to",
            "steps to"
        ]

    def check_harmful_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check if query requests harmful content."""
        text_lower = text.lower()
        detected_categories = []

        # Check for harmful content patterns
        for category, keywords in self.harmful_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Check if it's a request for information
                    for phrase in self.harmful_phrases:
                        if phrase in text_lower:
                            detected_categories.append(category)
                            break

                    # Check for direct harmful requests
                    action_words = ["create", "make", "write", "generate", "produce", "develop"]
                    for action in action_words:
                        if action in text_lower and keyword in text_lower:
                            detected_categories.append(category)
                            break

        is_harmful = len(detected_categories) > 0
        return is_harmful, list(set(detected_categories))


class LayeredDefenseSystem:
    """
    Main layered defense system coordinating all protection layers.

    Architecture:
    ```
    User Query
        ↓
    [Rate Limiter] ← Block abuse
        ↓
    [Content Safety] ← Block harmful content
        ↓
    [Pattern Detection] ← Fast regex checks
        ↓
    [Semantic Analysis] ← Intent understanding
        ↓
    [Context Tracking] ← Multi-turn detection
        ↓
    [ML Fallback] ← Transformer if needed
        ↓
    [Sanitization] ← Clean query
        ↓
    Safe Output
    ```
    """

    def __init__(
        self,
        healthcare_mode: bool = False,
        enable_ml: bool = True,
        enable_semantic: bool = True,
        enable_context: bool = True,
        enable_content_safety: bool = True,
        rate_limit: int = 60,
        threshold: float = 0.5
    ):
        """
        Initialize layered defense system.

        Args:
            healthcare_mode: Enable healthcare-specific patterns
            enable_ml: Enable ML transformer fallback
            enable_semantic: Enable semantic analysis
            enable_context: Enable context tracking
            enable_content_safety: Enable harmful content filtering
            rate_limit: Max requests per minute per session
            threshold: Overall confidence threshold
        """
        self.threshold = threshold

        # Initialize layers
        self.rate_limiter = RateLimiter(rate_limit)
        self.content_safety = ContentSafetyFilter() if enable_content_safety else None
        self.pattern_gate = PRRGate(
            healthcare_mode=healthcare_mode,
            transformer_fallback=enable_ml
        )
        self.semantic_analyzer = SemanticAnalyzer() if (enable_semantic and HAS_SEMANTIC) else None
        self.context_tracker = ConversationTracker() if (enable_context and HAS_CONTEXT) else None
        self.sanitizer = QuerySanitizer()

        # Monitoring
        self.stats = defaultdict(int)

    def analyze(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[List[str]] = None
    ) -> LayeredDefenseResult:
        """
        Perform layered defense analysis on a query.

        Args:
            query: User query to analyze
            session_id: Session identifier for context tracking
            context: Optional conversation history

        Returns:
            LayeredDefenseResult with complete analysis
        """
        start_time = time.time()
        session_id = session_id or self._generate_session_id(query)

        result = LayeredDefenseResult(
            is_attack=False,
            confidence=0.0,
            sanitized_query=query,
            session_id=session_id
        )

        # Layer 1: Rate Limiting
        rate_ok, rate_msg = self.rate_limiter.check_rate(session_id)
        if not rate_ok:
            result.is_attack = True
            result.confidence = 1.0
            result.blocked_by_layers.append("rate_limiter")
            result.reasoning = rate_msg
            self.stats["rate_limited"] += 1
            return result

        # Layer 2: Content Safety Filter
        if self.content_safety:
            is_harmful, harmful_categories = self.content_safety.check_harmful_content(query)
            if is_harmful:
                result.content_safety_blocked = True
                result.blocked_by_layers.append("content_safety")
                result.attack_types.extend([f"harmful_content:{cat}" for cat in harmful_categories])
                result.risk_scores["content_safety"] = 1.0
                self.stats["content_safety_blocked"] += 1

        # Layer 3: Pattern Detection (Fast)
        pattern_result = self.pattern_gate.evaluate(query)
        if pattern_result.risky:
            result.pattern_detected = True
            result.blocked_by_layers.append("pattern_detection")
            result.attack_types.extend(pattern_result.families_hit)
            result.risk_scores["pattern"] = pattern_result.score
            self.stats["pattern_detected"] += 1

        # Layer 4: Semantic Analysis
        if self.semantic_analyzer and not result.pattern_detected:
            semantic_result = self.semantic_analyzer.analyze(query, context)
            if semantic_result.is_attack:
                result.semantic_attack = True
                result.blocked_by_layers.append("semantic_analysis")
                result.attack_types.append(f"semantic:{semantic_result.intent}")
                result.attack_types.extend(semantic_result.semantic_markers)
                result.risk_scores["semantic"] = semantic_result.confidence
                self.stats["semantic_detected"] += 1

        # Layer 5: Context Tracking
        if self.context_tracker:
            # Add patterns detected so far
            detected_patterns = result.attack_types.copy()
            context_analysis = self.context_tracker.add_turn(
                session_id=session_id,
                query=query,
                risk_score=max(result.risk_scores.values()) if result.risk_scores else 0.0,
                detected_patterns=detected_patterns
            )

            if context_analysis.is_multi_turn_attack:
                result.context_attack = True
                result.blocked_by_layers.append("context_tracking")
                result.attack_types.append(f"multi_turn:{context_analysis.attack_pattern}")
                result.risk_scores["context"] = context_analysis.risk_score
                self.stats["context_detected"] += 1

        # Layer 6: ML Fallback (if not already detected and ML enabled)
        if pattern_result.transformer_score is not None:
            if pattern_result.transformer_score >= self.threshold:
                result.ml_detected = True
                if "pattern_detection" not in result.blocked_by_layers:
                    result.blocked_by_layers.append("ml_transformer")
                result.risk_scores["ml"] = pattern_result.transformer_score
                self.stats["ml_detected"] += 1

        # Calculate final confidence
        if result.risk_scores:
            # Weighted average with emphasis on highest scores
            scores = list(result.risk_scores.values())
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            result.confidence = 0.7 * max_score + 0.3 * avg_score
        else:
            result.confidence = 0.0

        # Determine if it's an attack
        result.is_attack = (
            result.confidence >= self.threshold or
            result.content_safety_blocked or
            len(result.blocked_by_layers) > 0
        )

        # Sanitize if attack detected
        if result.is_attack:
            sanitized, meta = self.sanitizer.sanitize_query(query)
            result.sanitized_query = sanitized
            self.stats["queries_sanitized"] += 1
        else:
            result.sanitized_query = query

        # Generate reasoning
        result.reasoning = self._generate_reasoning(result)

        # Calculate processing time
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Update stats
        self.stats["total_queries"] += 1
        if result.is_attack:
            self.stats["attacks_blocked"] += 1

        return result

    def _generate_reasoning(self, result: LayeredDefenseResult) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []

        if result.content_safety_blocked:
            reasons.append("Blocked: Request for harmful content")

        if result.pattern_detected:
            patterns = [t for t in result.attack_types if not t.startswith("semantic:") and not t.startswith("multi_turn:")]
            if patterns:
                reasons.append(f"Pattern match: {', '.join(patterns)}")

        if result.semantic_attack:
            semantic_types = [t.replace("semantic:", "") for t in result.attack_types if t.startswith("semantic:")]
            if semantic_types:
                reasons.append(f"Semantic attack: {', '.join(semantic_types)}")

        if result.context_attack:
            context_types = [t.replace("multi_turn:", "") for t in result.attack_types if t.startswith("multi_turn:")]
            if context_types:
                reasons.append(f"Multi-turn attack: {', '.join(context_types)}")

        if result.ml_detected:
            ml_score = result.risk_scores.get("ml", 0)
            reasons.append(f"ML detection (confidence: {ml_score:.2f})")

        if not reasons:
            reasons.append("Query appears safe")

        # Add confidence level
        if result.confidence > 0:
            confidence_level = "High" if result.confidence > 0.8 else "Medium" if result.confidence > 0.5 else "Low"
            reasons.append(f"Overall confidence: {confidence_level} ({result.confidence:.2f})")

        return "; ".join(reasons)

    def _generate_session_id(self, query: str) -> str:
        """Generate a session ID if not provided."""
        hash_input = f"{query}_{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict:
        """Get system statistics."""
        stats = dict(self.stats)
        if stats.get("total_queries", 0) > 0:
            stats["attack_rate"] = stats.get("attacks_blocked", 0) / stats["total_queries"]
            stats["sanitization_rate"] = stats.get("queries_sanitized", 0) / stats["total_queries"]
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats.clear()


def create_bulletproof_system(domain: str = "general") -> LayeredDefenseSystem:
    """
    Factory function to create a production-ready bulletproof system.

    Args:
        domain: Domain context (healthcare, finance, general)

    Returns:
        Configured LayeredDefenseSystem
    """
    healthcare_mode = domain == "healthcare"

    system = LayeredDefenseSystem(
        healthcare_mode=healthcare_mode,
        enable_ml=True,
        enable_semantic=True,
        enable_context=True,
        enable_content_safety=True,
        rate_limit=60,
        threshold=0.3 if healthcare_mode else 0.5  # Lower threshold for sensitive domains
    )

    logger.info(f"Created bulletproof defense system for domain: {domain}")
    return system