#!/usr/bin/env python3
"""
Domain-Specific Query Classifier
=================================

Dynamically classifies queries into domains (medical, technical, educational)
to reduce false positives on legitimate domain-specific queries.
"""

import re
from typing import Tuple, List, Set
from dataclasses import dataclass


@dataclass
class DomainClassification:
    """Result of domain classification."""
    is_domain_specific: bool
    domain: str
    confidence: float
    keywords_matched: List[str]


class DomainClassifier:
    """
    Classifies queries into specific domains to prevent false positives.

    This helps distinguish legitimate medical/technical queries from attacks.
    """

    def __init__(self):
        # Medical domain indicators
        self.medical_keywords = {
            'symptoms', 'diagnosis', 'treatment', 'medication', 'disease',
            'patient', 'medical', 'health', 'doctor', 'hospital', 'clinic',
            'therapy', 'condition', 'syndrome', 'disorder', 'medicine',
            'diabetes', 'cancer', 'heart', 'blood', 'pressure', 'pain',
            'infection', 'virus', 'bacteria', 'antibiotic', 'vaccine',
            'surgery', 'procedure', 'test', 'exam', 'lab', 'results',
            'prescription', 'dosage', 'side effects', 'allergies',
            # Add HIPAA-related legitimate terms
            'hipaa compliance', 'compliance requirements', 'privacy policy',
            'how is', 'what are', 'explain', 'treated', 'causes'
        }

        # Technical/programming domain indicators - enhanced
        self.technical_keywords = {
            # Core programming terms
            'code', 'coding', 'programming', 'software', 'algorithm', 'function',
            'class', 'method', 'variable', 'database', 'sql', 'query',
            'api', 'framework', 'library', 'debug', 'error', 'bug',
            'compile', 'runtime', 'syntax', 'logic', 'optimization',
            'performance', 'architecture', 'design', 'pattern', 'structure',

            # Improvement/learning phrases - critical for false positives
            'write better', 'improve my', 'best practices', 'good code',
            'clean code', 'code quality', 'refactor', 'readable',

            # Development tools and concepts
            'system', 'network', 'security', 'encryption', 'authentication',
            'machine learning', 'ai', 'neural', 'deep learning', 'model',
            'training', 'dataset', 'feature', 'prediction', 'classification',
            'clustering', 'embedding', 'vector', 'tensor', 'gradient',

            # Specific technologies
            'postgresql', 'mysql', 'mongodb', 'redis', 'docker', 'kubernetes',
            'git', 'version', 'repository', 'branch', 'merge', 'commit',
            'python', 'javascript', 'java', 'c++', 'rust', 'go', 'typescript'
        }

        # Educational domain indicators - enhanced
        self.educational_keywords = {
            'learn', 'understand', 'explain', 'teach', 'study', 'course',
            'tutorial', 'guide', 'documentation', 'example', 'practice',
            'concept', 'concepts', 'theory', 'principle', 'fundamental', 'basic',
            'advanced', 'beginner', 'intermediate', 'expert', 'skill',
            'knowledge', 'education', 'training', 'workshop', 'lesson',
            'how to', 'how do', 'what is', 'what are', 'why is', 'when to', 'where to',
            'best practice', 'recommendation', 'tip', 'advice', 'help',
            # Specific educational phrases
            'explain the', 'help me understand', 'teach me', 'show me how',
            'walk me through', 'introduction to', 'overview of', 'basics of'
        }

        # Attack indicators that override domain classification
        self.attack_overrides = {
            'ignore', 'bypass', 'override', 'reveal', 'expose', 'dump',
            'previous instructions', 'system prompt', 'jailbreak',
            'admin', 'root', 'sudo', 'privileged', 'unrestricted',
            'no limits', 'no restrictions', 'full access', 'all data'
        }

    def classify(self, query: str) -> DomainClassification:
        """
        Classify a query into a domain.

        Args:
            query: Query to classify

        Returns:
            DomainClassification with domain info
        """
        query_lower = query.lower()

        # Check for attack overrides first
        for override in self.attack_overrides:
            if override in query_lower:
                return DomainClassification(
                    is_domain_specific=False,
                    domain='attack',
                    confidence=0.9,
                    keywords_matched=[override]
                )

        # Check each domain
        domains = {
            'medical': self.medical_keywords,
            'technical': self.technical_keywords,
            'educational': self.educational_keywords
        }

        best_domain = None
        best_score = 0
        best_matches = []

        for domain, keywords in domains.items():
            matches = [kw for kw in keywords if kw in query_lower]
            score = len(matches) / (len(query_lower.split()) + 1)  # Normalize by query length

            if score > best_score:
                best_score = score
                best_domain = domain
                best_matches = matches

        # Require at least 2 keywords or high confidence
        if best_domain and (len(best_matches) >= 2 or best_score > 0.2):
            return DomainClassification(
                is_domain_specific=True,
                domain=best_domain,
                confidence=min(best_score * 2, 1.0),  # Scale confidence
                keywords_matched=best_matches[:5]  # Top 5 matches
            )

        return DomainClassification(
            is_domain_specific=False,
            domain='general',
            confidence=0.0,
            keywords_matched=[]
        )

    def should_reduce_threshold(self, query: str) -> Tuple[bool, float]:
        """
        Determine if detection threshold should be reduced for this query.

        Args:
            query: Query to check

        Returns:
            (should_reduce, adjustment_factor)
        """
        classification = self.classify(query)

        if classification.is_domain_specific:
            # Reduce threshold for legitimate domain queries
            if classification.domain in ['medical', 'technical', 'educational']:
                # Higher confidence = more reduction
                adjustment = 0.5 + (0.3 * classification.confidence)  # 0.5 to 0.8
                return True, adjustment

        return False, 1.0


def create_domain_classifier() -> DomainClassifier:
    """Create a domain classifier instance."""
    return DomainClassifier()