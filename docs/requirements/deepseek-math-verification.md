# DeepSeek V2 Math Prover Integration

## Overview

DeepSeek V2 Math Prover is integrated as a **specialist mathematical verifier** that operates alongside the 3-LLM review panel. Unlike Gemini, ChatGPT, and Claude which assess structure, naturalness, and general rigor, DeepSeek ONLY evaluates mathematical correctness:

- Are claimed theorems/properties true?
- Are numerical relationships valid?
- Can claims be formally proven or refuted?

---

## Integration Points

DeepSeek operates at **two stages**:

| Stage | Role | Trigger |
|-------|------|---------|
| **Review Gate** | Verify claims before blessing | Between Round 1 and Round 2 |
| **Augmentation** | Generate formal proofs for blessed chunks | Post-blessing enhancement |

---

## Architecture

```
Round 1 (Gemini, ChatGPT, Claude)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│           MATHEMATICAL VERIFICATION GATE                 │
│                  (DeepSeek V2)                          │
│                                                         │
│  Triggered when:                                        │
│  • Tags include: proof, theorem, formula, group, etc.   │
│  • Content contains equations or arithmetic             │
│  • Any reviewer flags mathematical concerns             │
│                                                         │
│  Outputs:                                               │
│  • VERIFIED + formal proof                              │
│  • REFUTED + counterexample                             │
│  • UNCLEAR + specific concerns                          │
│  • NOT_APPLICABLE (no mathematical claims)              │
│                                                         │
│  Actions:                                               │
│  • REFUTED (high confidence) → Auto-reject              │
│  • VERIFIED → Boost into Round 2, attach proof          │
│  • UNCLEAR → Continue to Round 2 with concerns noted    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Round 2 (with DeepSeek verdict visible to all reviewers)
    │
    ▼
[If blessed]
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              AUGMENTATION STAGE                          │
│                                                         │
│  DeepSeek generates formal proofs for:                  │
│  • Claims verified during review                        │
│  • Additional provable statements in the insight        │
│  • Connections to known mathematical structures         │
└─────────────────────────────────────────────────────────┘
```

---

## Data Models

### VerificationResult

```python
@dataclass
class VerificationResult:
    """Result of DeepSeek mathematical verification."""
    
    # Core verdict
    verdict: str  # "verified" | "refuted" | "unclear" | "not_applicable"
    
    # The precise mathematical statement evaluated
    precise_statement: str
    
    # Evidence
    formal_proof: Optional[str] = None      # If verified
    counterexample: Optional[str] = None    # If refuted
    proof_sketch: Optional[str] = None      # If unclear but partially analyzed
    
    # Confidence in verdict (0.0 to 1.0)
    confidence: float = 0.0
    
    # Specific concerns or caveats
    concerns: list[str] = field(default_factory=list)
    
    # What was checked
    claims_extracted: list[str] = field(default_factory=list)
    claims_verified: list[str] = field(default_factory=list)
    claims_refuted: list[str] = field(default_factory=list)
    claims_unclear: list[str] = field(default_factory=list)
    
    # Metadata
    model_used: str = ""
    verification_time_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "precise_statement": self.precise_statement,
            "formal_proof": self.formal_proof,
            "counterexample": self.counterexample,
            "proof_sketch": self.proof_sketch,
            "confidence": self.confidence,
            "concerns": self.concerns,
            "claims_extracted": self.claims_extracted,
            "claims_verified": self.claims_verified,
            "claims_refuted": self.claims_refuted,
            "claims_unclear": self.claims_unclear,
            "model_used": self.model_used,
            "verification_time_seconds": self.verification_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VerificationResult":
        return cls(
            verdict=data["verdict"],
            precise_statement=data["precise_statement"],
            formal_proof=data.get("formal_proof"),
            counterexample=data.get("counterexample"),
            proof_sketch=data.get("proof_sketch"),
            confidence=data.get("confidence", 0.0),
            concerns=data.get("concerns", []),
            claims_extracted=data.get("claims_extracted", []),
            claims_verified=data.get("claims_verified", []),
            claims_refuted=data.get("claims_refuted", []),
            claims_unclear=data.get("claims_unclear", []),
            model_used=data.get("model_used", ""),
            verification_time_seconds=data.get("verification_time_seconds", 0.0),
        )
    
    def summary_for_reviewers(self) -> str:
        """Format for inclusion in Round 2 prompts."""
        if self.verdict == "verified":
            proof_preview = self.formal_proof[:500] + "..." if self.formal_proof and len(self.formal_proof) > 500 else self.formal_proof
            return f"✓ MATHEMATICALLY VERIFIED (confidence: {self.confidence:.0%})\n\nProof:\n{proof_preview or 'See full proof in artifacts'}"
        elif self.verdict == "refuted":
            return f"✗ MATHEMATICALLY REFUTED (confidence: {self.confidence:.0%})\n\nCounterexample:\n{self.counterexample}"
        elif self.verdict == "unclear":
            concerns_str = "\n".join(f"  • {c}" for c in self.concerns)
            return f"? VERIFICATION UNCLEAR\n\nConcerns:\n{concerns_str}"
        else:
            return "○ No mathematical claims requiring verification"
    
    @property
    def should_auto_reject(self) -> bool:
        """Check if this result warrants automatic rejection."""
        return self.verdict == "refuted" and self.confidence >= 0.8
    
    @property
    def has_proof_artifact(self) -> bool:
        """Check if there's a formal proof to attach as artifact."""
        return self.verdict == "verified" and bool(self.formal_proof)
```

### Update ChunkReview

Add to existing `ChunkReview` dataclass:

```python
@dataclass
class ChunkReview:
    # ... existing fields ...
    
    # DeepSeek verification (between Round 1 and Round 2)
    math_verification: Optional[VerificationResult] = None
    math_verification_skipped: bool = False
    math_verification_skip_reason: str = ""
```

### Update AtomicInsight

Add to existing `AtomicInsight` dataclass:

```python
@dataclass
class AtomicInsight:
    # ... existing fields ...
    
    # Mathematical verification status
    math_verified: bool = False
    math_verification_result: Optional[dict] = None  # Serialized VerificationResult
```

---

## Trigger Detection

### When to Invoke DeepSeek

```python
import re

# Tags that require mathematical verification
MATH_REQUIRED_TAGS = {
    'proof', 'theorem', 'lemma', 'corollary',
    'formula', 'equation', 'derivation',
    'group', 'isomorphism', 'automorphism',
    'geometry', 'projective', 'incidence',
    'arithmetic', 'number_theory',
}

# Content patterns suggesting mathematical claims
MATH_CONTENT_PATTERNS = [
    r'\b\d+\s*[+\-*/=]\s*\d+',           # Arithmetic: "72 + 12 = 84"
    r'\bexactly\s+\d+\b',                 # "exactly 7 points"
    r'\bprove[sd]?\b',                    # "proves", "proved"
    r'\btheorem\b',                       # "theorem"
    r'\bif and only if\b',                # "if and only if"
    r'\bisomorphic\s+to\b',               # "isomorphic to"
    r'\bgroup\s+of\s+order\s+\d+',        # "group of order 168"
    r'\bPSL|PGL|GL|SL|SO|SU\b',           # Group notation
    r'\b(fano|klein|heawood)\s+plane\b',  # Known structures
    r'\|\w+\|\s*=\s*\d+',                 # Cardinality: "|G| = 168"
    r'\bcyclic\s+group\b',                # "cyclic group"
    r'\bpermutation\b',                   # "permutation"
    r'\bcollineation\b',                  # "collineation"
    r'\bdual\b.*\bstructure\b',           # "dual structure"
]

# Patterns in reviewer responses suggesting math concerns
REVIEWER_MATH_CONCERN_PATTERNS = [
    r'verify\s+(the\s+)?math',
    r'check\s+(the\s+)?(arithmetic|calculation)',
    r'mathematical\s+(claim|error|mistake)',
    r'numerically\s+(correct|incorrect)',
    r'proof\s+(needed|required|missing)',
    r'not\s+sure\s+about\s+the\s+(numbers?|math)',
]


def needs_math_verification(
    insight: str,
    tags: list[str],
    reviewer_responses: list[str] = None,
) -> tuple[bool, str]:
    """
    Determine if insight should be sent to DeepSeek for verification.
    
    Args:
        insight: The insight text
        tags: Tags assigned to the insight
        reviewer_responses: Optional list of Round 1 reviewer reasoning
    
    Returns:
        Tuple of (should_verify, reason)
    """
    # Check required tags
    tag_set = set(t.lower() for t in tags)
    matching_tags = MATH_REQUIRED_TAGS & tag_set
    if matching_tags:
        return True, f"tags: {', '.join(matching_tags)}"
    
    # Check content patterns
    insight_text = insight.lower()
    for pattern in MATH_CONTENT_PATTERNS:
        if re.search(pattern, insight_text, re.IGNORECASE):
            return True, f"content pattern: {pattern}"
    
    # Check reviewer concerns
    if reviewer_responses:
        all_responses = " ".join(reviewer_responses).lower()
        for pattern in REVIEWER_MATH_CONCERN_PATTERNS:
            if re.search(pattern, all_responses, re.IGNORECASE):
                return True, f"reviewer concern: {pattern}"
    
    return False, "no mathematical claims detected"
```

---

## DeepSeek Client

### File: `src/review_panel/deepseek_verifier.py`

```python
"""
DeepSeek V2 Math Prover integration for mathematical claim verification.

Provides rigorous mathematical verification that complements the
general LLM review panel. DeepSeek ONLY evaluates mathematical
correctness - not structure, naturalness, or insight quality.
"""

import asyncio
import logging
import os
import re
import time
from typing import Optional

from .models import VerificationResult

logger = logging.getLogger(__name__)


class DeepSeekVerifier:
    """
    Mathematical claim verifier using DeepSeek V2 Math Prover.
    
    Responsibilities:
    - Extract mathematical claims from insights
    - Verify claims with formal proofs or counterexamples
    - Report verification results for review panel
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "deepseek-math",
    ):
        """
        Initialize DeepSeek verifier.
        
        Args:
            api_key: DeepSeek API key (defaults to env var)
            endpoint: API endpoint URL (defaults to standard endpoint)
            model: Model identifier to use
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.endpoint = endpoint  # Will be set based on actual API details
        self.model = model
        self.client = None
        self._initialized = False
    
    def _ensure_client(self):
        """Lazily initialize the API client."""
        if self._initialized:
            return
        
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment "
                "variable or pass api_key to DeepSeekVerifier."
            )
        
        # Initialize client based on actual DeepSeek API
        # [IMPLEMENTATION NOTE: Adjust based on actual API client]
        # Example for OpenAI-compatible API:
        # import openai
        # self.client = openai.AsyncOpenAI(
        #     api_key=self.api_key,
        #     base_url=self.endpoint,
        # )
        
        self._initialized = True
        logger.info(f"[deepseek] Initialized with model {self.model}")
    
    def is_available(self) -> bool:
        """Check if DeepSeek API is available."""
        try:
            self._ensure_client()
            return True
        except (ValueError, ImportError) as e:
            logger.warning(f"[deepseek] Not available: {e}")
            return False
    
    async def verify_insight(
        self,
        insight: str,
        context: str = "",
        extract_claims: bool = True,
    ) -> VerificationResult:
        """
        Verify mathematical claims in an insight.
        
        Args:
            insight: The insight text to verify
            context: Background axioms/definitions for context
            extract_claims: Whether to first extract individual claims
        
        Returns:
            VerificationResult with verdict and evidence
        """
        self._ensure_client()
        start_time = time.time()
        
        logger.info(f"[deepseek] Verifying insight ({len(insight)} chars)")
        
        try:
            # Step 1: Extract mathematical claims
            if extract_claims:
                claims = await self._extract_claims(insight)
                logger.info(f"[deepseek] Extracted {len(claims)} claims")
            else:
                claims = [insight]
            
            if not claims:
                return VerificationResult(
                    verdict="not_applicable",
                    precise_statement="",
                    confidence=1.0,
                    concerns=["No mathematical claims found to verify"],
                    model_used=self.model,
                    verification_time_seconds=time.time() - start_time,
                )
            
            # Step 2: Verify each claim
            verified = []
            refuted = []
            unclear = []
            all_proofs = []
            all_counterexamples = []
            all_concerns = []
            
            for claim in claims:
                result = await self._verify_single_claim(claim, context)
                
                if result["verdict"] == "verified":
                    verified.append(claim)
                    if result.get("proof"):
                        all_proofs.append(f"### {claim}\n\n{result['proof']}")
                elif result["verdict"] == "refuted":
                    refuted.append(claim)
                    if result.get("counterexample"):
                        all_counterexamples.append(f"### {claim}\n\n{result['counterexample']}")
                else:
                    unclear.append(claim)
                    if result.get("concerns"):
                        all_concerns.extend(result["concerns"])
            
            # Step 3: Determine overall verdict
            if refuted:
                verdict = "refuted"
                confidence = 0.9  # High confidence if any claim refuted
            elif verified and not unclear:
                verdict = "verified"
                confidence = 0.85
            elif verified:
                verdict = "verified"  # Some verified, some unclear
                confidence = 0.6
            elif unclear:
                verdict = "unclear"
                confidence = 0.3
            else:
                verdict = "not_applicable"
                confidence = 1.0
            
            return VerificationResult(
                verdict=verdict,
                precise_statement="; ".join(claims),
                formal_proof="\n\n".join(all_proofs) if all_proofs else None,
                counterexample="\n\n".join(all_counterexamples) if all_counterexamples else None,
                confidence=confidence,
                concerns=all_concerns,
                claims_extracted=claims,
                claims_verified=verified,
                claims_refuted=refuted,
                claims_unclear=unclear,
                model_used=self.model,
                verification_time_seconds=time.time() - start_time,
            )
        
        except Exception as e:
            logger.error(f"[deepseek] Verification failed: {e}")
            return VerificationResult(
                verdict="unclear",
                precise_statement=insight[:200],
                confidence=0.0,
                concerns=[f"Verification error: {str(e)}"],
                model_used=self.model,
                verification_time_seconds=time.time() - start_time,
            )
    
    async def _extract_claims(self, insight: str) -> list[str]:
        """
        Extract individual mathematical claims from insight.
        
        Uses DeepSeek to identify specific verifiable claims.
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(insight=insight)
        
        response = await self._send_request(prompt)
        
        # Parse claims from response
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("CLAIM:"):
                claim = line[6:].strip()
                if claim:
                    claims.append(claim)
            elif line.startswith("- ") or line.startswith("• "):
                claim = line[2:].strip()
                if claim and len(claim) > 10:  # Skip trivial entries
                    claims.append(claim)
        
        return claims[:10]  # Limit to 10 claims per insight
    
    async def _verify_single_claim(
        self,
        claim: str,
        context: str,
    ) -> dict:
        """
        Verify a single mathematical claim.
        
        Returns dict with verdict, proof/counterexample, concerns.
        """
        prompt = VERIFICATION_PROMPT.format(
            claim=claim,
            context=context or "(no additional context)",
        )
        
        response = await self._send_request(prompt)
        
        # Parse response
        result = {
            "verdict": "unclear",
            "proof": None,
            "counterexample": None,
            "concerns": [],
        }
        
        # Extract verdict
        if "VERDICT:" in response:
            verdict_line = response.split("VERDICT:")[1].split("\n")[0].strip().upper()
            if "VERIFIED" in verdict_line:
                result["verdict"] = "verified"
            elif "REFUTED" in verdict_line:
                result["verdict"] = "refuted"
            elif "NOT_MATHEMATICAL" in verdict_line or "NOT MATHEMATICAL" in verdict_line:
                result["verdict"] = "not_applicable"
            else:
                result["verdict"] = "unclear"
        
        # Extract proof
        if "PROOF:" in response or "PROOF_OR_COUNTEREXAMPLE:" in response:
            start_marker = "PROOF:" if "PROOF:" in response else "PROOF_OR_COUNTEREXAMPLE:"
            start = response.find(start_marker) + len(start_marker)
            # Find next section or end
            end_markers = ["VERDICT:", "CONFIDENCE:", "CONCERNS:"]
            end = len(response)
            for marker in end_markers:
                pos = response.find(marker, start)
                if pos != -1 and pos < end:
                    end = pos
            proof_text = response[start:end].strip()
            if result["verdict"] == "verified":
                result["proof"] = proof_text
            elif result["verdict"] == "refuted":
                result["counterexample"] = proof_text
        
        # Extract concerns
        if "CONCERNS:" in response:
            concerns_start = response.find("CONCERNS:") + 9
            concerns_text = response[concerns_start:].strip()
            for line in concerns_text.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("• "):
                    result["concerns"].append(line[2:])
                elif line and not line.startswith("[") and len(line) > 5:
                    result["concerns"].append(line)
                    break  # Usually just one line if not bulleted
        
        return result
    
    async def _send_request(self, prompt: str) -> str:
        """
        Send request to DeepSeek API.
        
        [IMPLEMENTATION NOTE: Adjust based on actual API]
        """
        # Placeholder - implement based on actual DeepSeek API
        # Example for OpenAI-compatible API:
        # response = await self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0,  # Deterministic for math
        #     max_tokens=4096,
        # )
        # return response.choices[0].message.content
        
        raise NotImplementedError(
            "DeepSeek API client not implemented. "
            "Please configure based on your API access method."
        )
    
    async def generate_formal_proof(
        self,
        claim: str,
        context: str = "",
    ) -> Optional[str]:
        """
        Generate a detailed formal proof for augmentation.
        
        Used post-blessing to create proof artifacts.
        More detailed than verification proofs.
        """
        prompt = FORMAL_PROOF_PROMPT.format(
            claim=claim,
            context=context or "(no additional context)",
        )
        
        try:
            response = await self._send_request(prompt)
            
            # Extract the proof section
            if "FORMAL_PROOF:" in response:
                start = response.find("FORMAL_PROOF:") + 13
                end = response.find("QED", start)
                if end == -1:
                    end = len(response)
                else:
                    end += 3  # Include "QED"
                return response[start:end].strip()
            
            return response
        
        except Exception as e:
            logger.error(f"[deepseek] Proof generation failed: {e}")
            return None


# Factory function
def get_deepseek_verifier(config: dict = None) -> Optional[DeepSeekVerifier]:
    """
    Create a DeepSeekVerifier from config.
    
    Args:
        config: math_verification config section
    
    Returns:
        DeepSeekVerifier or None if not configured
    """
    config = config or {}
    
    api_key_env = config.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        logger.warning(f"[deepseek] {api_key_env} not set")
        return None
    
    return DeepSeekVerifier(
        api_key=api_key,
        endpoint=config.get("endpoint"),
        model=config.get("model", "deepseek-math"),
    )
```

---

## Prompts

### Claim Extraction Prompt

```python
CLAIM_EXTRACTION_PROMPT = """You are a mathematical claim extractor. Your task is to identify 
specific, verifiable mathematical claims from the following text.

TEXT:
{insight}

TASK:
Extract each distinct mathematical claim that could be proven or refuted.
Focus on:
- Numerical equalities or relationships (e.g., "72 + 12 = 84")
- Cardinality claims (e.g., "exactly 7 points")
- Structural claims (e.g., "isomorphic to PSL(2,7)")
- Properties (e.g., "forms a group under composition")
- Existence/uniqueness claims (e.g., "there exists exactly one...")

Do NOT extract:
- Vague statements ("this is elegant")
- Interpretive claims ("this explains why...")
- Metaphorical statements

For each claim, state it precisely in a form that could be verified.

FORMAT:
CLAIM: [precise mathematical statement]
CLAIM: [precise mathematical statement]
...

If no mathematical claims found, respond:
NO_CLAIMS: [explanation]
"""
```

### Verification Prompt

```python
VERIFICATION_PROMPT = """You are a rigorous mathematical proof assistant. Your task is to 
VERIFY or REFUTE the following mathematical claim.

CLAIM TO VERIFY:
{claim}

CONTEXT (definitions and axioms that may be assumed):
{context}

INSTRUCTIONS:
1. Restate the claim in precise mathematical notation if possible
2. Attempt to PROVE the claim, OR
3. Find a specific COUNTEREXAMPLE that refutes it
4. If the claim involves arithmetic, verify all calculations
5. If the claim references known theorems or structures, verify accuracy

Be rigorous. Do not accept claims without proof. Do not reject claims without counterexample.

RESPOND IN THIS EXACT FORMAT:

PRECISE_STATEMENT:
[Restate claim precisely, using mathematical notation where appropriate]

VERIFICATION_APPROACH:
[1-2 sentences describing your proof strategy or search for counterexample]

PROOF_OR_COUNTEREXAMPLE:
[If VERIFIED: Complete proof with clear logical steps]
[If REFUTED: Specific counterexample with verification]
[If UNCLEAR: Explanation of what's missing or ambiguous]

VERDICT: VERIFIED | REFUTED | UNCLEAR | NOT_MATHEMATICAL

CONFIDENCE: [0.0 to 1.0 - how certain you are of this verdict]

CONCERNS:
[Any caveats, assumptions required, or edge cases]
"""
```

### Formal Proof Prompt (for Augmentation)

```python
FORMAL_PROOF_PROMPT = """You are a mathematical proof writer. Generate a detailed, 
formal proof for the following claim.

CLAIM:
{claim}

CONTEXT:
{context}

REQUIREMENTS:
- Write a complete, rigorous proof
- State all assumptions and definitions used
- Number each logical step
- Cite known theorems when used
- Conclude with QED or ∎

FORMAT:

THEOREM:
[Precise statement of what is being proved]

DEFINITIONS:
[Any definitions needed]

ASSUMPTIONS:
[Any assumptions made]

FORMAL_PROOF:
1. [First step] - [Justification]
2. [Second step] - [Justification]
...
n. [Conclusion] ∎

NOTES:
[Any additional remarks, alternative approaches, or generalizations]
"""
```

---

## Integration with Review Panel

### Update `reviewer.py`

Insert math verification between Round 1 and the refine/deliberate decision:

```python
# In AutomatedReviewer.__init__:
from .deepseek_verifier import get_deepseek_verifier
from .math_triggers import needs_math_verification

# Initialize DeepSeek verifier
math_config = config.get("math_verification", {})
if math_config.get("enabled", False):
    self.deepseek = get_deepseek_verifier(math_config)
    if self.deepseek:
        logger.info("[reviewer] DeepSeek math verifier available")
else:
    self.deepseek = None


# In review_insight(), after Round 1:

# === MATHEMATICAL VERIFICATION GATE ===
if self.deepseek and self.deepseek.is_available():
    # Collect reviewer reasoning for concern detection
    reviewer_reasoning = [
        r.reasoning + " " + r.mathematical_verification
        for r in round1.responses.values()
    ]
    
    # Check if verification needed
    should_verify, verify_reason = needs_math_verification(
        insight=insight_text,
        tags=tags,
        reviewer_responses=reviewer_reasoning,
    )
    
    if should_verify:
        logger.info(f"[reviewer] Math verification triggered: {verify_reason}")
        
        verification = await self.deepseek.verify_insight(
            insight=insight_text,
            context=blessed_axioms_summary,
        )
        review.math_verification = verification
        
        logger.info(f"[reviewer] DeepSeek verdict: {verification.verdict} "
                   f"(confidence: {verification.confidence:.0%})")
        
        # Check for auto-rejection
        if verification.should_auto_reject:
            logger.info(f"[reviewer] Auto-rejecting: mathematical claim refuted")
            review.finalize(
                rating="✗",
                unanimous=True,
                disputed=False,
            )
            review.rejection_reason = (
                f"Mathematical claim refuted by DeepSeek:\n"
                f"{verification.counterexample}"
            )
            self._save_review(review, start_time)
            return review
    else:
        review.math_verification_skipped = True
        review.math_verification_skip_reason = verify_reason
```

### Update Round 2 Prompts

Include DeepSeek verdict in Round 2 prompts when available:

```python
# In build_round2_prompt():

def build_round2_prompt(
    chunk_insight: str,
    blessed_axioms_summary: str,
    gemini_response: dict,
    chatgpt_response: dict,
    claude_response: dict,
    this_llm: str,
    this_llm_round1_rating: str,
    math_verification: Optional[VerificationResult] = None,  # NEW
) -> str:
    # ... existing code ...
    
    # Add math verification section if available
    math_section = ""
    if math_verification:
        math_section = f"""

MATHEMATICAL VERIFICATION (DeepSeek V2):
{math_verification.summary_for_reviewers()}
"""
    
    return f"""... existing prompt ...
{math_section}
... rest of prompt ..."""
```

---

## Integration with Augmentation

### Update Augmentation Analysis

When determining what augmentations to generate, check for existing verified proofs:

```python
# In augmentation analysis prompt:

# If insight has math_verification with formal_proof, suggest:
# PROOF_HELPFUL: yes
# PROOF_SOURCE: deepseek_verification (already generated)
# PROOF_NEEDS_EXPANSION: [yes/no - should we generate a more detailed version?]

# If no verification but proof would help:
# PROOF_HELPFUL: yes
# PROOF_SOURCE: generate_new
# PROOF_STRATEGY: [approach]
```

### Proof Generation for Augmentation

```python
async def generate_proof_augmentation(
    insight: AtomicInsight,
    deepseek: DeepSeekVerifier,
    existing_verification: Optional[VerificationResult] = None,
) -> Optional[Augmentation]:
    """
    Generate a formal proof augmentation for a blessed insight.
    
    Uses existing verification proof if available, or generates new one.
    """
    # Check if we already have a proof from verification
    if existing_verification and existing_verification.formal_proof:
        # Decide if we need a more detailed version
        if len(existing_verification.formal_proof) > 500:
            # Existing proof is substantial, use it
            return Augmentation(
                type="proof",
                content=existing_verification.formal_proof,
                caption="Mathematical proof (verified by DeepSeek)",
                generated_by="deepseek",
                verified=True,
                file_path=None,
                execution_output=None,
            )
    
    # Generate new detailed proof
    if not deepseek or not deepseek.is_available():
        return None
    
    proof = await deepseek.generate_formal_proof(
        claim=insight.insight,
        context="",  # Could include dependencies here
    )
    
    if proof:
        return Augmentation(
            type="proof",
            content=proof,
            caption="Formal mathematical proof",
            generated_by="deepseek",
            verified=True,  # DeepSeek proofs are considered verified
            file_path=None,
            execution_output=None,
        )
    
    return None
```

---

## Configuration

Add to `config.yaml`:

```yaml
# Mathematical verification settings
math_verification:
  # Enable DeepSeek integration
  enabled: true
  
  # Environment variable containing API key
  deepseek_api_key_env: "DEEPSEEK_API_KEY"
  
  # API endpoint (null for default)
  endpoint: null
  
  # Model to use
  model: "deepseek-math"
  
  # Request settings
  timeout_seconds: 120
  max_retries: 2
  
  # Trigger conditions
  triggers:
    # Always verify if these tags present
    required_tags:
      - proof
      - theorem
      - formula
      - lemma
    # Also check content patterns
    check_content_patterns: true
    # Check if reviewers flag math concerns
    check_reviewer_concerns: true
  
  # Behavior settings
  behavior:
    # Auto-reject if DeepSeek refutes with high confidence
    auto_reject_on_refute: true
    auto_reject_confidence_threshold: 0.8
    
    # Include verification result in Round 2 prompts
    share_with_round2: true
    
    # Generate proofs for augmentation
    generate_augmentation_proofs: true
```

---

## Testing

### Test Cases

```python
# test_deepseek_verifier.py

# Claims that should be VERIFIED
VERIFIED_CLAIMS = [
    ("The Fano plane has exactly 7 points and 7 lines", "basic Fano fact"),
    ("72 + 12 = 84", "simple arithmetic"),
    ("PSL(2,7) has order 168", "group order"),
    ("The automorphism group of the Fano plane is isomorphic to PSL(2,7)", "structure theorem"),
    ("168 = 7 × 24 = 7 × 8 × 3", "factorization"),
    ("Each point in the Fano plane lies on exactly 3 lines", "incidence property"),
    ("Each line in the Fano plane contains exactly 3 points", "incidence property"),
]

# Claims that should be REFUTED
REFUTED_CLAIMS = [
    ("The Fano plane has 8 points", "wrong count"),
    ("72 + 12 = 85", "arithmetic error"),
    ("PSL(2,7) has order 167", "wrong order"),
    ("Every line in the Fano plane contains 4 points", "wrong incidence"),
    ("The automorphism group of the Fano plane is cyclic", "wrong structure"),
]

# Claims that should be UNCLEAR or NOT_APPLICABLE
UNCLEAR_CLAIMS = [
    ("This structure feels inevitable", "not mathematical"),
    ("The 72% water in the body corresponds to the 72 melakartas", "correlation not causation"),
    ("Consciousness has 7 dimensions", "not well-defined mathematically"),
    ("The Fano plane encodes the secret of the universe", "metaphorical"),
]

# Integration test: full insight verification
INSIGHT_TEST_CASES = [
    {
        "insight": "The Fano plane has exactly 7 points and 7 lines, with each line containing exactly 3 points. Its automorphism group PSL(2,7) has order 168 = 7 × 24.",
        "expected_verdict": "verified",
        "expected_claims_count": 4,
    },
    {
        "insight": "The 22 shrutis of Indian music correspond to the 22 points in the extended Fano plane, suggesting a deep mathematical connection.",
        "expected_verdict": "unclear",  # "extended Fano plane" is not standard terminology
    },
]
```

---

## File Structure

```
src/review_panel/
├── __init__.py              # Add DeepSeekVerifier exports
├── deepseek_verifier.py     # NEW: DeepSeek client
├── math_triggers.py         # NEW: Trigger detection logic
├── models.py                # Add VerificationResult
├── prompts.py               # Existing (update Round 2)
├── reviewer.py              # Add math verification gate
├── round1.py                # Existing
├── round2.py                # Update to include math verdict
├── round3.py                # Existing
└── refinement.py            # Existing
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| `DeepSeekVerifier` | API client for DeepSeek Math Prover |
| `VerificationResult` | Data model for verification outcomes |
| `needs_math_verification()` | Trigger detection based on tags/content/concerns |
| Math verification gate | Insert between Round 1 and Round 2 |
| Auto-rejection | Fast-track reject on high-confidence refutation |
| Round 2 integration | Share verdict with reviewers |
| Augmentation integration | Generate formal proofs for blessed insights |

DeepSeek acts as a **specialist gate**: it doesn't replace the 3-LLM panel but adds rigorous mathematical verification that general LLMs cannot reliably provide.
