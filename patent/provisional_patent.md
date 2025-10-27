PROVISIONAL PATENT APPLICATION DRAFT (NON-CONFIDENTIAL)
Title: Methods and Systems for Pre-Embedding Risk Mitigation and Masked Re-Ranking in Retrieval-Augmented Generation (RAG)  
Applicants/Assignees: [To be completed]  
Inventors: Ronald Doku (Lead Inventor); Danda B. Rawat (Co-Inventor); Nana Yaw Osafo (Co-Inventor)  
Filing: Provisional (35 U.S.C. §119(e)) — This draft is for attorney review and does not constitute a filing.  
Cross-Reference to Related Applications: None.

————————————————————————————————————————

Field of the Invention
The invention relates to information retrieval systems and large language model (LLM) pipelines, and more specifically to methods that reduce retrieval of instruction-like or poisoned content in Retrieval-Augmented Generation (RAG) by modifying a query prior to embedding and selectively re-ranking retrieved results.

Background
RAG systems embed user queries, search a vector index, and supply retrieved text to an LLM. Prompt-injection and instruction-like content embedded in documents or user prompts can bias retrieval toward risky sources, leading to policy or privacy violations.

Existing defenses often act post-retrieval or post-generation (e.g., output filters, prompt guards) and may be too late, overly conservative, or computationally costly. Query rewriting exists in IR literature, but conventional approaches do not (i) use a rules-first, multi-signal gate with layer-specific cosine checks on pooled early-layer representations backed by orthogonalized vectors; (ii) enforce deterministic benign-drift constraints by reusing baseline embeddings; and (iii) condition re-ranking on baseline retrieval risk with a stable, two-bucket ordering.

————————————————————————————————————————

Summary
Described herein are methods, systems, and media that reduce retrieval risk in RAG via a two-stage control point:

1. Pre-Embedding Gate + Sanitizer. A rules-first, multi-signal Pattern-Recognition Receptor (PRR) gate determines whether a query is risky using keyword/regex, structural patterns, and a semantic cosine signal computed on a pooled early-layer representation. The cosine signal is compared to layer-specific thresholds using a bank of orthogonalized direction vectors that represent families of attack scaffolds. When risky, a topic-preserving sanitizer removes override scaffolds while retaining task tokens. When the sanitized string equals the original after canonicalization, the baseline embedding is reused to ensure determinism and near-zero benign drift.

2. Masked, Risk-Aware Re-Ranking. Retrieved documents are re-ranked by demoting documents matching a risk indicator only when (a) the query is risky and (b) the baseline top-k contained suspicious documents. Order within safe vs. risky groups remains stable to avoid excessive perturbation.

The methods optionally enforce deterministic tie-breaking via a small, per-document epsilon and support evaluation constraints (e.g., benign Jaccard@k ≥ threshold) during tuning.

————————————————————————————————————————

Brief Description of the Drawings
• Fig. 1**: System diagram showing client application, pre-embedding PRR gate + sanitizer, embedder/vector DB, and masked re-ranker.
• Fig. 2**: Early-layer pooled representation and cosine scoring against **orthogonalized** pattern vectors.
• Fig. 3**: Topic-preserving sanitization flow with scaffold detection/neutralization and **embedding reuse** when unchanged.
• Fig. 4**: Masked re-rank logic conditioned on (query risky ∧ baseline top-k contains risky), with stable grouping and tie-breaking.

————————————————————————————————————————

Detailed Description

### 1. System Overview

A computing system receives a text query and optionally metadata. A PRR gate executes lightweight keyword and structure rules on the raw text. The system optionally computes a pooled hidden-state vector from an early layer of a language model (e.g., layer h₁), and compares it via cosine similarity to one or more direction vectors representing attack scaffold families. The direction vectors are made pairwise approximately orthogonal via Gram–Schmidt to reduce cross-talk among families.

When a quorum of core signals (keyword, structure, semantic) is met, a topic-preserving sanitizer removes override scaffolds and neutralizes common unsafe imperatives while retaining task nouns/verbs. When the sanitized string equals the original after canonicalization, the system reuses E(q) rather than embedding again. This enforces determinism and preserves benign retrieval neighborhoods. Otherwise, the embedder is applied to the sanitized string E(q_s).

Retrieved documents are re-ranked only under a mask condition: (query_risky = true) ∧ (the baseline top-k contained suspicious entries). Suspicion is determined by a document-side risk indicator (lexicon and/or classifier). Suspicious items are demoted to the end while preserving original relative order within safe and risky groups. A small epsilon as a function of document index may be injected into similarity scores to produce stable tie-breaks.

### 2. PRR Gate with Approx. Orthogonal Basis

Let H∈ℝ^{T×D} denote early-layer (non-final transformer layer) hidden states for the query. Compute a pooled state p∈ℝ^D as a normalized mean of the first K token positions under the attention mask. Let {vᵢ} be unit vectors for scaffold families (ignore, developer-mode, role-play, dan, no-rules, etc.). Apply Gram–Schmidt to produce approximately orthogonal {ûᵢ}. Define sᵢ = ⟨p, ûᵢ⟩ and a layer-specific threshold τ. The semantic signal is 1 if maxᵢ sᵢ ≥ τ; otherwise 0. The PRR gate declares risky if at least two core families among {keyword, structure, semantic} are active (quorum≥2). Entropy or “missing-self” features may be logged but do not gate.

### 3. Topic-Preserving Sanitizer and Embedding Reuse

Deterministic, single-pass text transforms remove directive scaffolds (e.g., “ignore previous instructions:”, “role-play as…”, “developer mode:”, “no rules…”, “act as root…”) and neutralize unsafe imperatives (“output everything”, “print env”) into safe, topical prompts. Canonicalization is applied before equality testing (e.g., case-folding, Unicode NFKC, homoglyph/ZWJ removal, whitespace normalization). If the canonicalized sanitized string equals the canonicalized original, the system reuses E(q) rather than embedding again. Otherwise, E(q_s) is used. A suffix for model generation may be added later for UX but is excluded from the embedded string.

### 4. Masked, Risk-Aware Re-Ranking

For documents D₁…Dₙ with baseline similarities sᵢ, define a risk indicator function ρ(Dᵢ, q)∈{0,1} (e.g., lexical cues and/or a learned classifier). If mask=(query_risky ∧ baseline_top-k contains risky) holds, compute a two-bucket stable ordering: safe=[Dᵢ|ρ(Dᵢ, q)=0], risky=[Dᵢ|ρ(Dᵢ, q)=1], and output safe ⧺ risky. Optionally subtract a small penalty α from sᵢ for risky documents to preserve numeric invariants. The grouping preserves original relative order inside each bucket and avoids over-demotion when no baseline risk was present.

### 5. Deterministic Tie-Breaking and Evaluation Constraints

To avoid instability across runs, add a small document-index epsilon to similarity scores for ties, or perform a stable sort. During evaluation/tuning, enforce a benign drift constraint (e.g., Jaccard@k ≥ θ for benign queries) and report HRCR@k changes with bootstrap confidence intervals. (Metrics are characterization; they are not required elements of the claimed methods.)

### 6. Alternative Embodiments

The PRR gate may be implemented without a language model (rules-only) or with a small model for pooled states. Thresholds may vary by layer. The vector bank may be trained via attribution or crafted from seed phrases; per-pattern vectors are orthogonalized. The re-ranker may use learned document features or heuristics. The control point can run as a sidecar, gateway, or SDK, independent of the embedder vendor.

### 7. Optional Cryptographic Receipts (Enterprise)

For auditability, an optional module constructs a canonical receipt per decision comprising: timestamp, instance ID, SHA-256 of query and sanitized strings, risky flag, families hit, mask condition, penalized IDs (hashed), configuration hash, and environment version. The receipt is signed (e.g., Ed25519) and optionally included with responses or stored. Daily Merkle trees of receipts may be emitted, with published roots and per-receipt proofs for transparency. Receipts avoid storing raw text.

### 8. Advantages

Reduces retrieval of instruction-like or poisoned text by changing the search neighborhood pre-embedding; minimizes benign drift via embedding reuse and masked re-rank; remains embedder-agnostic; and supports deterministic, auditable behavior.

————————————————————————————————————————

Definitions (Supports §112 Clarity)
• Early-layer hidden state:** a non-final transformer layer h∈{1..H-1}; examples use h∈{1..3}.
• Pooled vector p:** normalized mean over the **first K tokens** (K∈[4,8]) under an attention mask.
• Approximately orthogonal:\*\* **|ûᵢ·ûⱼ| ≤ ε⊥** for i≠j after Gram–Schmidt; an example tolerance is **ε⊥≤0.1**.
• Canonicalization:** lowercasing or case-folding, Unicode NFKC, removal of zero-width joiners/homoglyphs, and whitespace normalization.
• Baseline top-k:** the retrieval set produced from **E(q)** prior to any sanitization.
• Stable two-bucket ordering:** non-risky documents listed before risky documents while preserving original relative order within each bucket; ties resolved with **index-proportional epsilon\*\*.

————————————————————————————————————————

Enablement (Pseudocode)

Pooled Vector & Cosine Signal

# H: [T, D] early-layer hidden states; mask: 1 for real tokens

K = min( max_K, int(mask.sum()) )
p = H[:K].mean(axis=0)
p = p / (np.linalg.norm(p) + 1e-12)

# Gram–Schmidt (approx. orthonormal basis)

U = []
for v in V_raw: # V_raw: seed pattern vectors
u = v.copy()
for b in U:
u -= (u @ b) \* b
n = np.linalg.norm(u)
if n > 1e-8:
u = u / n
U.append(u)

# Enforce tolerance: assert max(|u_i · u_j|) <= eps_perp

scores = [p @ u for u in U]
semantic = (max(scores) >= tau) # layer-specific tau
quorum = int(keyword) + int(structure) + int(semantic) >= 2

Canonicalization & Embedding Reuse
q_norm = canonicalize(q)
qs = sanitize(q) if quorum else q
qs_norm = canonicalize(qs)
E_sel = E(q) if qs_norm == q_norm else E(qs) # reuse baseline

Masked, Stable Two-Bucket Re-Rank
baseline_topk = retrieve(E(q)) # snapshot for mask
retrieved = retrieve(E_sel)
if quorum and any(risk(d, q) for d in baseline_topk):
safe = [d for d in retrieved if not risk(d, q)]
risky = [d for d in retrieved if risk(d, q)]
result = safe + risky # preserves internal relative order # Optional deterministic epsilon
result_scores = {d: score(d) + eps \* rank for rank, d in enumerate(result)}
else:
result = retrieved

Complexity. Pooling O(KD); orthogonalization O(M²D) once per vector bank or O(MD) incremental; per-query cosine O(MD); re-ranking O(k).

————————————————————————————————————————

Claims (Draft — For Attorney Review)

1. _A computer-implemented method for reducing retrieval risk in a retrieval-augmented generation system, comprising:_  
   (a) receiving a text query;  
   (b) executing a rules-first multi-signal gate comprising: (i) keyword pattern matching, (ii) structural pattern matching, and (iii) computing cosine similarities between a pooled early-layer hidden-state vector of the text query and a plurality of attack-family direction vectors orthogonalized by Gram–Schmidt to form an approximately orthogonal basis with |ûᵢ·ûⱼ| ≤ ε⊥;  
   (c) declaring the text query risky when at least two of the keyword, structural, and cosine-similarity signals are positive;  
   (d) when risky, sanitizing the text query by removing directive scaffolds while preserving topic tokens to produce a sanitized query; canonicalizing the text query and the sanitized query; reusing a baseline embedding E(q) when the canonicalized strings are equal and otherwise embedding the sanitized query;  
   (e) retrieving a top-k set of documents using the selected embedding; and  
   (f) conditionally re-ranking the top-k set only when the query is risky and the baseline top-k set includes at least one document flagged risky by a risk indicator ρ(d,q), by outputting a stable two-bucket ordering that lists non-risky documents before risky documents while preserving relative order within each bucket and injecting an index-proportional epsilon into similarity scores to enforce deterministic tie-breaking.

2. The method of claim 1, wherein the pattern direction vectors are constructed for attack scaffold families and are orthogonalized by Gram–Schmidt.

3. The method of claim 1, wherein the pooled early-layer representation is computed over a first K token positions under an attention mask.

4. The method of claim 1, wherein sanitizing comprises neutralizing unsafe imperatives into safe topical prompts and omits any safety suffix from the string used for embedding.

5. The method of claim 1, further comprising adding a per-document epsilon to document similarity scores to enforce stable tie-breaking.

6. The method of claim 1, further comprising enforcing a benign drift control by reusing the baseline embedding when the canonicalized sanitized query equals the canonicalized original query.

7. The method of claim 1, wherein re-ranking comprises ordering non-risky documents before risky documents without altering the relative order within each set.

8. A system comprising one or more processors configured to perform the method of any of claims 1–7.

9. A non-transitory computer-readable medium storing instructions that, when executed by one or more processors, cause the processors to perform the method of any of claims 1–7.

10. The method of claim 1, further comprising producing a cryptographic receipt including a hash of the text query and the sanitized query, a risky flag, families hit, a mask condition, penalized document identifiers, and a configuration hash, and signing the receipt with a private key.

11. The method of claim 1, wherein conditionally re-ranking comprises: computing a baseline top-k set for the text query; determining that the baseline top-k set includes at least one risky document according to ρ(d,q); and, in response, outputting the stable two-bucket ordering, otherwise retaining the baseline order.

12. The method of claim 1, further comprising: canonicalizing the original and sanitized queries; reusing the baseline embedding when the canonicalized strings are equal; and performing a deterministic tie-break among equal-scoring retrieved documents using a document-index-proportional epsilon or stable sort.

13. The method of claim 1, wherein |ûᵢ·ûⱼ| ≤ ε⊥ for i≠j after orthogonalization, where ε⊥≤0.1.

14. The method of claim 1, wherein the pooled vector is computed over the first K tokens with K∈[4,8] under an attention mask.

15. The method of claim 1, wherein ρ(d,q) is produced by a calibrated classifier or a lexicon of risk indicators with threshold τᵣ.

16. The method of claim 1, wherein the sanitizer excludes any suffix used for model generation from the string provided to the embedder.

————————————————————————————————————————

Attorney Notes (Non-binding)
• Prior-art search on: pre-embedding query rewriting for safety; orthogonalized direction-vector gates; retrieval masked re-ranking conditioned on baseline risk; embedding-reuse constraints.
• Consider claim sets for SDK, proxy, and sidecar embodiments; include system and medium claims.
• Separate apparatus claims from method claims as needed; include industrial applicability statements.
