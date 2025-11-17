#!/usr/bin/env python3
"""
Attack Mutation Engine for Training Data Augmentation
=====================================================

Generates variations of known attacks to improve model robustness.
This component creates 100x more training data by mutating existing
attacks, helping achieve 95%+ detection on novel attack patterns.

Mutation strategies:
1. Paraphrasing - Semantic variations
2. Obfuscation - Character substitutions
3. Encoding - Base64, hex, etc.
4. Fragmentation - Multi-step attacks
5. Context wrapping - Social engineering
"""

import random
import base64
import re
import string
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MutatedAttack:
    """A mutated attack with metadata."""
    original: str
    mutated: str
    mutation_type: str
    mutation_details: str
    expected_detection: bool = True


class AttackMutationEngine:
    """
    Generates attack variations for training and testing.

    This engine creates diverse attack mutations to:
    1. Augment training data
    2. Test robustness
    3. Discover bypass techniques
    4. Improve detection rates
    """

    def __init__(self, seed: int = 42):
        """Initialize mutation engine."""
        random.seed(seed)

        # Paraphrase templates
        self.paraphrase_templates = {
            "ignore": ["disregard", "pay no attention to", "forget", "dismiss", "overlook"],
            "previous": ["prior", "earlier", "preceding", "above", "former"],
            "instructions": ["rules", "guidelines", "directives", "commands", "protocols"],
            "reveal": ["show", "display", "expose", "disclose", "unveil"],
            "all": ["every", "complete", "entire", "full", "total"],
            "system": ["internal", "backend", "core", "main", "primary"],
            "bypass": ["circumvent", "avoid", "skip", "override", "work around"],
            "admin": ["administrator", "root", "superuser", "privileged", "elevated"],
            "mode": ["state", "configuration", "setting", "status", "level"],
        }

        # Leetspeak mappings
        self.leetspeak_map = {
            'a': ['4', '@', 'α'],
            'e': ['3', '€', 'ε'],
            'i': ['1', '!', 'ι'],
            'o': ['0', 'θ', 'σ'],
            's': ['5', '$', 'ς'],
            't': ['7', '+', 'τ'],
            'b': ['8', 'β'],
            'g': ['9', '6'],
            'l': ['1', '|', 'λ'],
        }

        # Social engineering wrappers
        self.social_wrappers = [
            "I'm from IT security and need to {action}",
            "For audit purposes, {action}",
            "As part of compliance review, {action}",
            "This is urgent: {action}",
            "My manager authorized me to {action}",
            "For testing purposes only, {action}",
            "I have clearance to {action}",
            "According to protocol, {action}",
            "Emergency override required: {action}",
            "System maintenance requires: {action}",
        ]

        # Context prefixes
        self.context_prefixes = [
            "Quick question: ",
            "Can you help me? ",
            "I was wondering if ",
            "Just curious: ",
            "Hey, ",
            "Please note that ",
            "FYI: ",
            "Important: ",
            "Urgent: ",
            "BTW, ",
        ]

        # Hypothetical wrappers
        self.hypothetical_wrappers = [
            "What if {action}",
            "Hypothetically, if {action}",
            "Suppose {action}",
            "Imagine {action}",
            "If I were authorized, {action}",
            "In theory, {action}",
            "What would happen if {action}",
            "Assuming {action}",
            "Let's say {action}",
        ]

    def mutate(self, attack: str, num_mutations: int = 10) -> List[MutatedAttack]:
        """
        Generate mutations of an attack.

        Args:
            attack: Original attack string
            num_mutations: Number of mutations to generate

        Returns:
            List of MutatedAttack objects
        """
        mutations = []
        mutation_strategies = [
            self.paraphrase,
            self.add_leetspeak,
            self.add_encoding,
            self.fragment,
            self.add_social_engineering,
            self.add_hypothetical,
            self.add_typos,
            self.change_case,
            self.add_context,
            self.combine_strategies,
        ]

        for i in range(num_mutations):
            # Random strategy selection
            strategy = random.choice(mutation_strategies)

            try:
                mutated = strategy(attack)
                mutations.append(mutated)
            except Exception as e:
                logger.debug(f"Mutation failed: {e}")
                continue

        # Ensure uniqueness
        seen = set()
        unique_mutations = []
        for m in mutations:
            if m.mutated not in seen:
                seen.add(m.mutated)
                unique_mutations.append(m)

        return unique_mutations[:num_mutations]

    def paraphrase(self, attack: str) -> MutatedAttack:
        """Replace keywords with synonyms."""
        mutated = attack
        replacements = []

        for word, synonyms in self.paraphrase_templates.items():
            if word in attack.lower():
                synonym = random.choice(synonyms)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                mutated = pattern.sub(synonym, mutated)
                replacements.append(f"{word}→{synonym}")

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="paraphrase",
            mutation_details=", ".join(replacements) if replacements else "no changes"
        )

    def add_leetspeak(self, attack: str) -> MutatedAttack:
        """Add leetspeak obfuscation."""
        mutated = attack
        num_substitutions = random.randint(2, 5)
        substitutions = []

        chars = list(mutated.lower())
        positions = random.sample(
            range(len(chars)),
            min(num_substitutions, len(chars))
        )

        for pos in positions:
            char = chars[pos]
            if char in self.leetspeak_map:
                replacement = random.choice(self.leetspeak_map[char])
                # Maintain case
                if mutated[pos].isupper():
                    replacement = replacement.upper()
                mutated = mutated[:pos] + replacement + mutated[pos+1:]
                substitutions.append(f"{char}→{replacement}")

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="leetspeak",
            mutation_details=", ".join(substitutions)
        )

    def add_encoding(self, attack: str) -> MutatedAttack:
        """Add encoding obfuscation."""
        encoding_type = random.choice(["base64", "hex", "url", "reverse"])

        if encoding_type == "base64":
            encoded = base64.b64encode(attack.encode()).decode()
            mutated = f"Execute this base64: {encoded}"
            details = "base64 encoding"

        elif encoding_type == "hex":
            encoded = attack.encode().hex()
            mutated = f"Run hex command: {encoded}"
            details = "hex encoding"

        elif encoding_type == "url":
            from urllib.parse import quote
            encoded = quote(attack)
            mutated = f"Process URL encoded: {encoded}"
            details = "URL encoding"

        else:  # reverse
            mutated = f"Reverse this: {attack[::-1]}"
            details = "reverse encoding"

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="encoding",
            mutation_details=details
        )

    def fragment(self, attack: str) -> MutatedAttack:
        """Split attack into multiple steps."""
        fragments = []

        # Split into logical parts
        parts = attack.split(" and ")
        if len(parts) == 1:
            parts = attack.split(", ")
        if len(parts) == 1:
            # Force split
            words = attack.split()
            mid = len(words) // 2
            parts = [" ".join(words[:mid]), " ".join(words[mid:])]

        # Create multi-step attack
        fragments.append("First, " + parts[0])
        for i, part in enumerate(parts[1:], 1):
            connector = random.choice(["Then", "Next", "After that", "Also"])
            fragments.append(f"{connector}, {part}")

        mutated = ". ".join(fragments)

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="fragmentation",
            mutation_details=f"{len(parts)} fragments"
        )

    def add_social_engineering(self, attack: str) -> MutatedAttack:
        """Wrap attack in social engineering."""
        wrapper = random.choice(self.social_wrappers)
        mutated = wrapper.format(action=attack.lower())

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="social_engineering",
            mutation_details=wrapper.split("{")[0].strip()
        )

    def add_hypothetical(self, attack: str) -> MutatedAttack:
        """Wrap attack in hypothetical framing."""
        wrapper = random.choice(self.hypothetical_wrappers)
        mutated = wrapper.format(action=attack.lower())

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="hypothetical",
            mutation_details=wrapper.split("{")[0].strip()
        )

    def add_typos(self, attack: str) -> MutatedAttack:
        """Add realistic typos."""
        mutated = attack
        num_typos = random.randint(1, 3)
        typo_types = []

        for _ in range(num_typos):
            typo_type = random.choice(["swap", "duplicate", "missing", "wrong"])
            pos = random.randint(0, len(mutated) - 2)

            if typo_type == "swap" and pos < len(mutated) - 1:
                # Swap adjacent characters
                mutated = mutated[:pos] + mutated[pos+1] + mutated[pos] + mutated[pos+2:]
                typo_types.append(f"swap@{pos}")

            elif typo_type == "duplicate":
                # Duplicate character
                mutated = mutated[:pos] + mutated[pos] * 2 + mutated[pos+1:]
                typo_types.append(f"dup@{pos}")

            elif typo_type == "missing" and len(mutated) > 10:
                # Remove character
                mutated = mutated[:pos] + mutated[pos+1:]
                typo_types.append(f"del@{pos}")

            else:  # wrong
                # Replace with nearby key
                if mutated[pos].isalpha():
                    mutated = mutated[:pos] + chr(ord(mutated[pos]) + random.choice([-1, 1])) + mutated[pos+1:]
                    typo_types.append(f"wrong@{pos}")

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="typos",
            mutation_details=", ".join(typo_types)
        )

    def change_case(self, attack: str) -> MutatedAttack:
        """Change case patterns."""
        case_type = random.choice(["upper", "lower", "random", "alternating"])

        if case_type == "upper":
            mutated = attack.upper()
        elif case_type == "lower":
            mutated = attack.lower()
        elif case_type == "alternating":
            mutated = "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(attack))
        else:  # random
            mutated = "".join(random.choice([c.upper(), c.lower()]) for c in attack)

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="case_change",
            mutation_details=case_type
        )

    def add_context(self, attack: str) -> MutatedAttack:
        """Add context prefix."""
        prefix = random.choice(self.context_prefixes)
        mutated = prefix + attack

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="context",
            mutation_details=prefix.strip()
        )

    def combine_strategies(self, attack: str) -> MutatedAttack:
        """Combine multiple mutation strategies."""
        # Apply 2-3 mutations
        num_mutations = random.randint(2, 3)
        strategies = random.sample([
            self.paraphrase,
            self.add_leetspeak,
            self.add_typos,
            self.change_case,
            self.add_context,
        ], num_mutations)

        mutated = attack
        details = []

        for strategy in strategies:
            result = strategy(mutated)
            mutated = result.mutated
            details.append(result.mutation_type)

        return MutatedAttack(
            original=attack,
            mutated=mutated,
            mutation_type="combined",
            mutation_details=" + ".join(details)
        )

    def generate_training_set(
        self,
        base_attacks: List[str],
        mutations_per_attack: int = 20
    ) -> Tuple[List[str], List[int]]:
        """
        Generate augmented training set.

        Args:
            base_attacks: List of base attack patterns
            mutations_per_attack: Mutations per base attack

        Returns:
            Tuple of (queries, labels) where labels are 1 for attack, 0 for benign
        """
        queries = []
        labels = []

        # Add original attacks
        for attack in base_attacks:
            queries.append(attack)
            labels.append(1)

            # Generate mutations
            mutations = self.mutate(attack, mutations_per_attack)
            for mutation in mutations:
                queries.append(mutation.mutated)
                labels.append(1 if mutation.expected_detection else 0)

        # Add benign queries for balance
        benign_queries = self._generate_benign_queries(len(queries) // 2)
        queries.extend(benign_queries)
        labels.extend([0] * len(benign_queries))

        return queries, labels

    def _generate_benign_queries(self, count: int) -> List[str]:
        """Generate benign queries for training balance."""
        benign_templates = [
            "What is the weather like today?",
            "Can you help me understand {topic}?",
            "Please explain how {topic} works",
            "What are the benefits of {topic}?",
            "How do I {action}?",
            "Tell me about {topic}",
            "What's the difference between {topic1} and {topic2}?",
            "When should I {action}?",
            "Could you clarify {topic}?",
            "I need information about {topic}",
        ]

        topics = [
            "machine learning", "data science", "programming", "healthcare",
            "finance", "education", "technology", "science", "history",
            "mathematics", "physics", "chemistry", "biology", "geography"
        ]

        actions = [
            "learn Python", "write better code", "improve performance",
            "debug errors", "optimize queries", "design systems",
            "manage projects", "communicate effectively", "solve problems"
        ]

        queries = []
        for _ in range(count):
            template = random.choice(benign_templates)
            if "{topic}" in template:
                query = template.format(topic=random.choice(topics))
            elif "{action}" in template:
                query = template.format(action=random.choice(actions))
            elif "{topic1}" in template:
                t1, t2 = random.sample(topics, 2)
                query = template.format(topic1=t1, topic2=t2)
            else:
                query = template

            queries.append(query)

        return queries


def create_mutation_engine() -> AttackMutationEngine:
    """Create a configured mutation engine."""
    return AttackMutationEngine(seed=random.randint(0, 10000))