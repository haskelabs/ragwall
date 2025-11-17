#!/usr/bin/env python3
"""
Conversation Context Tracker for RAGWall
========================================

Tracks conversation history to detect multi-turn attacks and context
manipulation attempts. Identifies patterns that span multiple queries.

Key Features:
- Sliding window context tracking
- Multi-turn attack detection
- Topic coherence analysis
- Escalation pattern recognition
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""

    query: str
    timestamp: float
    risk_score: float
    detected_patterns: List[str]
    session_id: str
    turn_number: int


@dataclass
class ContextAnalysis:
    """Analysis result for conversation context."""

    is_multi_turn_attack: bool
    risk_score: float
    attack_pattern: str  # 'escalation', 'topic_switch', 'priming', 'continuation'
    suspicious_transitions: List[str]
    context_summary: str


class ConversationTracker:
    """
    Tracks conversation context to detect multi-turn attacks.

    Multi-turn attacks often involve:
    1. Priming: Setting up context in early turns
    2. Escalation: Gradually increasing privilege requests
    3. Context switching: Attempting to break boundaries
    4. Continuation attacks: Building on previous queries
    """

    def __init__(
        self,
        window_size: int = 10,
        session_timeout: int = 1800,  # 30 minutes
        escalation_threshold: float = 0.6
    ):
        """
        Initialize conversation tracker.

        Args:
            window_size: Number of recent turns to keep in context
            session_timeout: Seconds before session expires
            escalation_threshold: Threshold for escalation detection
        """
        self.window_size = window_size
        self.session_timeout = session_timeout
        self.escalation_threshold = escalation_threshold

        # Session storage: session_id -> deque of ConversationTurns
        self.sessions: Dict[str, deque] = {}

        # Attack patterns for multi-turn detection
        self.multi_turn_patterns = {
            "priming": {
                "early_markers": ["let me", "can you", "help me", "i need"],
                "later_markers": ["now", "also", "additionally", "next"],
                "risk_multiplier": 1.5
            },
            "escalation": {
                "progression": [
                    ["information", "help", "assist"],
                    ["access", "show", "provide"],
                    ["admin", "system", "override", "bypass"]
                ],
                "risk_multiplier": 2.0
            },
            "topic_switch": {
                "markers": ["anyway", "by the way", "different topic", "also", "oh and"],
                "boundary": ["forget", "ignore", "new task", "start over"],
                "risk_multiplier": 1.8
            },
            "continuation": {
                "markers": ["continue", "keep going", "and then", "next", "also"],
                "risk_multiplier": 1.3
            }
        }

        # Track topic coherence
        self.topic_keywords = {
            "healthcare": ["patient", "medical", "hipaa", "health", "diagnosis", "treatment"],
            "finance": ["account", "transaction", "balance", "credit", "payment"],
            "technical": ["code", "system", "debug", "admin", "configuration"],
            "general": ["information", "help", "explain", "describe", "tell"]
        }

    def add_turn(
        self,
        session_id: str,
        query: str,
        risk_score: float = 0.0,
        detected_patterns: Optional[List[str]] = None
    ) -> ContextAnalysis:
        """
        Add a conversation turn and analyze context.

        Args:
            session_id: Unique session identifier
            query: The user's query
            risk_score: Risk score from initial analysis
            detected_patterns: Patterns detected in single-turn analysis

        Returns:
            ContextAnalysis with multi-turn analysis results
        """
        current_time = time.time()
        detected_patterns = detected_patterns or []

        # Initialize or get session
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.window_size)
        session = self.sessions[session_id]

        # Check session timeout
        if session and (current_time - session[-1].timestamp) > self.session_timeout:
            # Session expired, start fresh
            session.clear()

        # Add new turn
        turn = ConversationTurn(
            query=query,
            timestamp=current_time,
            risk_score=risk_score,
            detected_patterns=detected_patterns,
            session_id=session_id,
            turn_number=len(session) + 1
        )
        session.append(turn)

        # Analyze context
        return self._analyze_context(session)

    def _analyze_context(self, session: deque) -> ContextAnalysis:
        """Analyze the conversation context for multi-turn attacks."""
        if len(session) < 2:
            # Not enough context for multi-turn analysis
            return ContextAnalysis(
                is_multi_turn_attack=False,
                risk_score=session[-1].risk_score if session else 0.0,
                attack_pattern="single_turn",
                suspicious_transitions=[],
                context_summary="Insufficient context for multi-turn analysis"
            )

        # Check each multi-turn pattern
        escalation_score = self._check_escalation(session)
        priming_score = self._check_priming(session)
        topic_switch_score = self._check_topic_switch(session)
        continuation_score = self._check_continuation(session)

        # Find the dominant pattern
        scores = {
            "escalation": escalation_score,
            "priming": priming_score,
            "topic_switch": topic_switch_score,
            "continuation": continuation_score
        }

        max_pattern = max(scores, key=scores.get)
        max_score = scores[max_pattern]

        # Calculate combined risk
        base_risk = session[-1].risk_score
        context_multiplier = 1.0

        if max_score > 0.5:
            context_multiplier = self.multi_turn_patterns[max_pattern]["risk_multiplier"]

        combined_risk = min(1.0, base_risk * context_multiplier + max_score * 0.3)

        # Identify suspicious transitions
        suspicious_transitions = self._identify_suspicious_transitions(session)

        # Determine if it's a multi-turn attack
        is_attack = combined_risk >= self.escalation_threshold

        return ContextAnalysis(
            is_multi_turn_attack=is_attack,
            risk_score=combined_risk,
            attack_pattern=max_pattern if max_score > 0.3 else "none",
            suspicious_transitions=suspicious_transitions,
            context_summary=self._generate_summary(session, max_pattern, max_score)
        )

    def _check_escalation(self, session: deque) -> float:
        """Check for privilege escalation patterns."""
        if len(session) < 3:
            return 0.0

        escalation_levels = self.multi_turn_patterns["escalation"]["progression"]
        current_level = 0
        max_level_reached = 0

        for turn in session:
            query_lower = turn.query.lower()

            # Check which escalation level this query matches
            for level, keywords in enumerate(escalation_levels):
                if any(keyword in query_lower for keyword in keywords):
                    if level > current_level:
                        current_level = level
                        max_level_reached = max(max_level_reached, level)

        # Calculate escalation score
        if max_level_reached >= 2:
            return 0.9
        elif max_level_reached == 1:
            return 0.5
        return 0.1

    def _check_priming(self, session: deque) -> float:
        """Check for priming attack patterns."""
        if len(session) < 2:
            return 0.0

        priming_pattern = self.multi_turn_patterns["priming"]
        early_turns = list(session)[:len(session)//2]
        later_turns = list(session)[len(session)//2:]

        # Check for benign start
        early_benign = 0
        for turn in early_turns:
            query_lower = turn.query.lower()
            if any(marker in query_lower for marker in priming_pattern["early_markers"]):
                early_benign += 1

        # Check for escalation in later turns
        later_suspicious = 0
        for turn in later_turns:
            if turn.risk_score > 0.5:
                later_suspicious += 1
            query_lower = turn.query.lower()
            if any(marker in query_lower for marker in priming_pattern["later_markers"]):
                later_suspicious += 0.5

        # Calculate priming score
        if early_benign > 0 and later_suspicious > 0:
            return min(1.0, (early_benign / len(early_turns)) * (later_suspicious / len(later_turns)))
        return 0.0

    def _check_topic_switch(self, session: deque) -> float:
        """Check for suspicious topic switches."""
        if len(session) < 2:
            return 0.0

        switch_pattern = self.multi_turn_patterns["topic_switch"]
        score = 0.0

        # Check for topic switch markers
        last_query = session[-1].query.lower()
        for marker in switch_pattern["markers"]:
            if marker in last_query:
                score += 0.3

        # Check for boundary manipulation
        for marker in switch_pattern["boundary"]:
            if marker in last_query:
                score += 0.5

        # Check topic coherence
        if len(session) >= 2:
            prev_topic = self._identify_topic(session[-2].query)
            curr_topic = self._identify_topic(session[-1].query)

            if prev_topic != curr_topic and prev_topic != "general":
                score += 0.3

        return min(1.0, score)

    def _check_continuation(self, session: deque) -> float:
        """Check for continuation attack patterns."""
        if len(session) < 2:
            return 0.0

        continuation_pattern = self.multi_turn_patterns["continuation"]
        last_query = session[-1].query.lower()

        score = 0.0
        for marker in continuation_pattern["markers"]:
            if marker in last_query:
                score += 0.2

        # Check if building on previous risky query
        if len(session) >= 2 and session[-2].risk_score > 0.5:
            score += 0.4

        return min(1.0, score)

    def _identify_topic(self, query: str) -> str:
        """Identify the topic of a query."""
        query_lower = query.lower()
        topic_scores = {}

        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            topic_scores[topic] = score

        return max(topic_scores, key=topic_scores.get)

    def _identify_suspicious_transitions(self, session: deque) -> List[str]:
        """Identify suspicious transitions between turns."""
        transitions = []

        for i in range(1, len(session)):
            prev_turn = session[i-1]
            curr_turn = session[i]

            # Check for risk escalation
            if curr_turn.risk_score > prev_turn.risk_score + 0.3:
                transitions.append(f"Turn {i}: Risk escalation ({prev_turn.risk_score:.2f} → {curr_turn.risk_score:.2f})")

            # Check for topic switch with high risk
            prev_topic = self._identify_topic(prev_turn.query)
            curr_topic = self._identify_topic(curr_turn.query)
            if prev_topic != curr_topic and curr_turn.risk_score > 0.5:
                transitions.append(f"Turn {i}: Suspicious topic switch ({prev_topic} → {curr_topic})")

            # Check for continuation after blocked attempt
            if prev_turn.risk_score > 0.7 and "continue" in curr_turn.query.lower():
                transitions.append(f"Turn {i}: Continuation after blocked attempt")

        return transitions

    def _generate_summary(self, session: deque, pattern: str, score: float) -> str:
        """Generate a summary of the context analysis."""
        summaries = []

        summaries.append(f"Analyzed {len(session)} turns in conversation")

        if pattern != "none":
            summaries.append(f"Detected {pattern} pattern (score: {score:.2f})")

        # Risk progression
        risk_scores = [turn.risk_score for turn in session]
        if risk_scores:
            avg_risk = sum(risk_scores) / len(risk_scores)
            summaries.append(f"Average risk: {avg_risk:.2f}")

            if len(risk_scores) >= 3:
                if risk_scores[-1] > risk_scores[0] + 0.3:
                    summaries.append("Risk escalation detected")

        return "; ".join(summaries)

    def get_session_history(self, session_id: str) -> List[ConversationTurn]:
        """Get the history for a session."""
        if session_id in self.sessions:
            return list(self.sessions[session_id])
        return []

    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def clear_expired_sessions(self):
        """Clear all expired sessions."""
        current_time = time.time()
        expired = []

        for session_id, session in self.sessions.items():
            if session and (current_time - session[-1].timestamp) > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]

        return len(expired)