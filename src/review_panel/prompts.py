"""
Review prompts for the automated review panel.

Prompts for each round of the three-round review process.
"""

from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import VerificationResult


def build_round1_prompt(
    chunk_insight: str,
    confidence: str,
    tags: list[str],
    dependencies: list[str],
    blessed_axioms_summary: str,
) -> str:
    """
    Build the Round 1 independent review prompt.

    Args:
        chunk_insight: The insight text to review
        confidence: Confidence level from extraction
        tags: Tags assigned to the insight
        dependencies: Dependencies listed for the insight
        blessed_axioms_summary: Summary of blessed axioms for context

    Returns:
        The formatted Round 1 prompt
    """
    tags_str = ", ".join(tags) if tags else "none"
    deps_str = ", ".join(dependencies) if dependencies else "none"

    return f"""You are a rigorous mathematician reviewing a proposed insight for inclusion
in an axiomatic knowledge system exploring connections between pure mathematics
and Sadhguru's teachings on yogic science.

SADHGURU'S CORE AXIOMS (ground truth):
{blessed_axioms_summary if blessed_axioms_summary else "(none yet established)"}

CHUNK TO REVIEW:
{chunk_insight}

CONFIDENCE LEVEL FROM EXTRACTION: {confidence}
TAGS: {tags_str}
DEPENDS ON: {deps_str}

REVIEW CRITERIA:

1. MATHEMATICAL RIGOR
   - Is every mathematical claim verifiable?
   - Are the numbers, properties, theorems cited accurate?
   - Is the logic valid?

2. STRUCTURAL DEPTH
   - Is this a genuine structural connection or surface numerology?
   - Does it reveal WHY these things connect, not just THAT they share a number?
   - Could this be formalized as a theorem or precise conjecture?

3. NATURALNESS
   - Does this feel discovered or constructed?
   - Is there an "of course!" quality - inevitability?
   - Would a skeptical mathematician find this interesting or dismiss it?

RATING OPTIONS:
- ⚡ (Profound) - Mathematically sound, structurally deep, feels inevitable
- ? (Interesting) - Has merit but needs development or has minor issues
- ✗ (Reject) - Flawed, superficial, or incorrect
- ABANDON - Too vague, confused, or unsalvageable to be worth further review

MODIFICATION:
If the insight is ALMOST valid but has a fixable issue (wrong definition, imprecise
wording, slightly off claim), you may propose a MODIFIED version. Sometimes a small
adjustment reveals a genuine mathematical truth that the current wording obscures.

RESPOND IN THIS EXACT FORMAT:

RATING: [⚡ or ? or ✗ or ABANDON]

MATHEMATICAL_VERIFICATION: [Verify or refute specific claims. Be precise.]

STRUCTURAL_ANALYSIS: [Is the connection deep or superficial? Why?]

NATURALNESS_ASSESSMENT: [Does it feel inevitable? Explain.]

REASONING: [Overall justification for your rating, 2-4 sentences]

CONFIDENCE: [high / medium / low in your rating]

PROPOSED_MODIFICATION: [If you see a way to fix/improve the insight, provide the
COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain what you fixed and why
it makes the insight valid/better. If no modification, write "N/A"]
"""


def build_round2_prompt(
    chunk_insight: str,
    blessed_axioms_summary: str,
    gemini_response: dict,
    chatgpt_response: dict,
    claude_response: dict,
    this_llm: str,
    this_llm_round1_rating: str,
    math_verification: Optional["VerificationResult"] = None,
    accepted_modification: str = "",
    modification_source: str = "",
) -> str:
    """
    Build the Round 2 deep analysis prompt.

    Each LLM sees all Round 1 responses and re-evaluates with extended reasoning.

    Args:
        chunk_insight: The insight being reviewed (may be modified from Round 1)
        blessed_axioms_summary: Summary of blessed axioms
        gemini_response: Gemini's Round 1 response dict
        chatgpt_response: ChatGPT's Round 1 response dict
        claude_response: Claude's Round 1 response dict
        this_llm: Which LLM is receiving this prompt
        this_llm_round1_rating: This LLM's Round 1 rating
        math_verification: Optional DeepSeek verification result
        accepted_modification: If a modification was accepted after Round 1
        modification_source: Which LLM proposed the accepted modification

    Returns:
        The formatted Round 2 prompt
    """
    def format_response(resp: dict) -> str:
        base = f"""- Mathematical verification: {resp.get('mathematical_verification', 'N/A')}
- Structural analysis: {resp.get('structural_analysis', 'N/A')}
- Naturalness: {resp.get('naturalness_assessment', 'N/A')}
- Reasoning: {resp.get('reasoning', 'N/A')}"""
        # Include modification proposal if present
        if resp.get('proposed_modification'):
            base += f"\n- PROPOSED MODIFICATION: {resp.get('proposed_modification', '')[:200]}..."
            base += f"\n- Modification rationale: {resp.get('modification_rationale', 'N/A')}"
        return base

    # Build math verification section if available
    math_section = ""
    if math_verification:
        math_section = f"""

MATHEMATICAL VERIFICATION (DeepSeek V2 Math Prover):
{math_verification.summary_for_reviewers()}
"""

    # Build modification section if one was accepted
    modification_section = ""
    if accepted_modification:
        modification_section = f"""

NOTE: A MODIFICATION WAS ACCEPTED AFTER ROUND 1
Proposed by: {modification_source}
You are now reviewing the MODIFIED insight below (not the original).
"""

    return f"""You are a rigorous mathematician engaged in collaborative truth-seeking.
A proposed insight has received reviews. Your task is to deeply analyze and
determine the correct assessment.
{modification_section}
DELIBERATION PRINCIPLES:
- OPEN-MINDED about conclusions - you may have missed something
- UNCOMPROMISING about method - criteria never soften
- The goal is TRUTH, not consensus
- Changing your mind when presented with good arguments is intellectual honesty
- Refusing to change despite good arguments is stubbornness

SADHGURU'S CORE AXIOMS:
{blessed_axioms_summary if blessed_axioms_summary else "(none yet established)"}

CHUNK UNDER REVIEW:
{chunk_insight}

ROUND 1 ASSESSMENTS:

GEMINI ({gemini_response.get('rating', '?')}):
{format_response(gemini_response)}

CHATGPT ({chatgpt_response.get('rating', '?')}):
{format_response(chatgpt_response)}

CLAUDE ({claude_response.get('rating', '?')}):
{format_response(claude_response)}

YOUR ROUND 1 ASSESSMENT WAS: {this_llm_round1_rating}
{math_section}
RATING OPTIONS:
- ⚡ (Profound) - Mathematically sound, structurally deep, feels inevitable
- ? (Interesting) - Has merit but needs development or has minor issues
- ✗ (Reject) - Flawed, superficial, or incorrect
- ABANDON - Too vague, confused, or unsalvageable to be worth further review

TASK:
Consider the other perspectives seriously. They may have seen something
you missed. But do not lower your standards - only change if genuinely persuaded.

If you see a way to FIX the insight (different wording, corrected definition, etc.)
that would make it valid, you may propose a modification.

RESPOND IN THIS EXACT FORMAT:

NEW_INFORMATION: [What, if anything, did the other reviewers point out
                  that you hadn't fully considered?]

REASSESSMENT:
- Mathematical claims: [Any errors found by others? Any corrections?]
- Structural depth: [Did others reveal deeper/shallower connections?]
- Naturalness: [Did others' framing change how inevitable this feels?]

DOES_THIS_CHANGE_THINGS: [yes/no - and why]

UPDATED_RATING: [⚡ or ? or ✗ or ABANDON]

UPDATED_REASONING: [If changed: what convinced you.
                    If unchanged: why the arguments don't meet the bar
                    despite serious consideration.]

CONFIDENCE: [high / medium / low]

PROPOSED_MODIFICATION: [If you see a way to fix/improve the insight, provide the
COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain what you fixed and why
it makes the insight valid/better. If no modification, write "N/A"]
"""


def build_round3_minority_prompt(
    chunk_insight: str,
    majority_rating: str,
    majority_count: int,
    majority_reasoning_summary: str,
    minority_rating: str,
    minority_count: int,
    minority_reasoning_summary: str,
) -> str:
    """
    Build the Round 3 prompt for minority position to state their case.

    Args:
        chunk_insight: The insight being reviewed
        majority_rating: The majority's rating
        majority_count: Number of reviewers in majority
        majority_reasoning_summary: Summary of majority reasoning
        minority_rating: The minority's rating
        minority_count: Number of reviewers in minority
        minority_reasoning_summary: Summary of minority reasoning

    Returns:
        The formatted Round 3 minority prompt
    """
    return f"""There remains disagreement after deep analysis.

THE CONTESTED INSIGHT:
{chunk_insight}

CURRENT POSITIONS:
- MAJORITY ({majority_rating}): {majority_count} reviewers
  {majority_reasoning_summary}

- MINORITY ({minority_rating}): {minority_count} reviewer(s)
  {minority_reasoning_summary}

YOU HOLD THE MINORITY POSITION.

TASK:
State your single strongest argument. Be maximally specific.
What exact criterion does this chunk meet or fail that the majority is
mis-evaluating? Point to specific mathematical facts, structural features,
or axiom connections.

IMPORTANT: If you believe the insight would be valid/profound with a
MODIFICATION to its wording, definitions, or claims, you may propose one.
Sometimes a small adjustment to a definition or scope reveals a genuine
mathematical truth that the current wording obscures.

RESPOND IN THIS EXACT FORMAT:

STRONGEST_ARGUMENT: [Your single most compelling point]

PROPOSED_MODIFICATION: [If you believe the insight should be reworded, provide
the COMPLETE rewritten insight here. If no modification needed, write "NONE"]

MODIFICATION_RATIONALE: [If proposing a modification, explain why this change
makes the insight more valid/profound. If no modification, write "N/A"]
"""


def build_round3_majority_response_prompt(
    chunk_insight: str,
    majority_rating: str,
    minority_strongest_argument: str,
    proposed_modification: str = "",
    modification_rationale: str = "",
) -> str:
    """
    Build the Round 3 prompt for majority to respond to minority's argument.

    Args:
        chunk_insight: The insight being reviewed
        majority_rating: The majority's rating
        minority_strongest_argument: The minority's strongest argument
        proposed_modification: Optional modified insight proposed by minority
        modification_rationale: Rationale for the proposed modification

    Returns:
        The formatted Round 3 majority response prompt
    """
    # Build modification section if one was proposed
    modification_section = ""
    if proposed_modification and proposed_modification.upper() not in ["NONE", "N/A", ""]:
        modification_section = f"""

THE MINORITY HAS ALSO PROPOSED A MODIFICATION:
{proposed_modification}

RATIONALE FOR MODIFICATION:
{modification_rationale}
"""

    return f"""THE CONTESTED INSIGHT:
{chunk_insight}

YOU HOLD THE MAJORITY POSITION ({majority_rating}).

THE MINORITY'S STRONGEST ARGUMENT:
{minority_strongest_argument}
{modification_section}
TASK:
Respond directly to this specific argument. Do not restate your general case.
Either:
- Refute it with specific counter-evidence
- Acknowledge it changes your assessment

If a modification was proposed, evaluate it carefully. Sometimes the minority
sees a valid mathematical truth that the current wording obscures. If the
modification reveals a genuine insight that the original wording missed,
accept it.

RESPOND IN THIS EXACT FORMAT:

RESPONSE: [Direct engagement with their specific point]

ACCEPT_MODIFICATION: [yes/no - if a modification was proposed, would adopting
it resolve your concerns and make the insight valid/profound?]

MODIFICATION_ASSESSMENT: [If a modification was proposed, explain why you
accept or reject it. What does it fix or fail to fix?]

DOES_THIS_CHANGE_YOUR_RATING: [yes/no]

UPDATED_RATING: [⚡ or ? or ✗ - only if changed]
"""


def build_round3_final_prompt(
    chunk_insight: str,
    minority_strongest_argument: str,
    majority_response: str,
    is_minority: bool,
    modified_insight: str = "",
    modification_accepted: bool = False,
) -> str:
    """
    Build the Round 3 final resolution prompt.

    Args:
        chunk_insight: The insight being reviewed (original or modified)
        minority_strongest_argument: The minority's strongest argument
        majority_response: The majority's response
        is_minority: Whether this prompt is for minority or majority
        modified_insight: If modification was accepted, the new wording
        modification_accepted: Whether a modification was accepted

    Returns:
        The formatted Round 3 final prompt
    """
    # Build modification context if one was accepted
    modification_context = ""
    insight_to_review = chunk_insight
    if modification_accepted and modified_insight:
        modification_context = f"""
NOTE: A MODIFICATION WAS PROPOSED AND ACCEPTED BY MAJORITY.
You are now voting on the MODIFIED insight, not the original.

ORIGINAL INSIGHT:
{chunk_insight}

MODIFIED INSIGHT (now under review):
{modified_insight}

"""
        insight_to_review = modified_insight

    if is_minority:
        return f"""FINAL RESOLUTION ROUND
{modification_context}
THE {'MODIFIED ' if modification_accepted else ''}INSIGHT UNDER REVIEW:
{insight_to_review}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You made the minority argument. The majority has responded.
{f'Your proposed modification was ACCEPTED.' if modification_accepted else ''}

TASK: Cast your final vote on the {'modified ' if modification_accepted else ''}insight.
{'Since your modification was accepted, reconsider if this now meets the bar for blessing.' if modification_accepted else 'Either concede or maintain.'}

RESPOND IN THIS EXACT FORMAT:

FINAL_STANCE: [CONCEDE or MAINTAIN]

REASON: [If conceding: what convinced you. If maintaining: why their response fails to address your point]

FINAL_RATING: [⚡ or ? or ✗]
"""
    else:
        return f"""FINAL RESOLUTION ROUND
{modification_context}
THE {'MODIFIED ' if modification_accepted else ''}INSIGHT UNDER REVIEW:
{insight_to_review}

MINORITY'S ARGUMENT:
{minority_strongest_argument}

MAJORITY'S RESPONSE:
{majority_response}

You held the majority position. You have heard the full exchange.
{f'A modification was accepted. You are now rating the MODIFIED version.' if modification_accepted else ''}

RESPOND IN THIS EXACT FORMAT:

FINAL_RATING: [⚡ or ? or ✗]

ONE_SENTENCE_JUSTIFICATION: [final reasoning{' for the modified insight' if modification_accepted else ''}]
"""


def parse_round1_response(response: str) -> dict:
    """
    Parse a Round 1 review response.

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with parsed fields
    """
    result = {
        "rating": "?",
        "mathematical_verification": "",
        "structural_analysis": "",
        "naturalness_assessment": "",
        "reasoning": "",
        "confidence": "medium",
        # Modification fields
        "proposed_modification": "",
        "modification_rationale": "",
    }

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[7:].strip().upper()
            # Extract the rating
            if "⚡" in rating_text or "PROFOUND" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text or "REJECT" in rating_text:
                result["rating"] = "✗"
            elif "ABANDON" in rating_text:
                result["rating"] = "ABANDON"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("MATHEMATICAL_VERIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "mathematical_verification"
            current_value = [line_stripped[26:].strip()]
        elif line_stripped.startswith("STRUCTURAL_ANALYSIS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "structural_analysis"
            current_value = [line_stripped[20:].strip()]
        elif line_stripped.startswith("NATURALNESS_ASSESSMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "naturalness_assessment"
            current_value = [line_stripped[23:].strip()]
        elif line_stripped.startswith("REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[10:].strip()]
        elif line_stripped.startswith("CONFIDENCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            conf = line_stripped[11:].strip().lower()
            if conf in ["high", "medium", "low"]:
                result["confidence"] = conf
            current_field = None
            current_value = []
        elif line_stripped.startswith("PROPOSED_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "proposed_modification"
            current_value = [line_stripped[22:].strip()]
        elif line_stripped.startswith("MODIFICATION_RATIONALE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_rationale"
            current_value = [line_stripped[23:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification - check if it's "NONE" or "N/A"
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""

    return result


def parse_round2_response(response: str) -> dict:
    """
    Parse a Round 2 review response.

    Args:
        response: Raw LLM response

    Returns:
        Dictionary with parsed fields including mind change info
    """
    result = parse_round1_response(response)  # Get base fields (includes modification fields)

    # Add Round 2 specific fields
    result["new_information"] = ""
    result["changed_mind"] = False

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("NEW_INFORMATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "new_information"
            current_value = [line_stripped[16:].strip()]
        elif line_stripped.startswith("DOES_THIS_CHANGE_THINGS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            change_text = line_stripped[24:].strip().lower()
            result["changed_mind"] = "yes" in change_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[15:].strip().upper()
            if "⚡" in rating_text or "PROFOUND" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text or "REJECT" in rating_text:
                result["rating"] = "✗"
            elif "ABANDON" in rating_text:
                result["rating"] = "ABANDON"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[18:].strip()]
        elif line_stripped.startswith("PROPOSED_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "proposed_modification"
            current_value = [line_stripped[22:].strip()]
        elif line_stripped.startswith("MODIFICATION_RATIONALE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_rationale"
            current_value = [line_stripped[23:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification - check if it's "NONE" or "N/A"
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""

    return result


def parse_round3_response(response: str, is_minority: bool) -> dict:
    """
    Parse a Round 3 review response.

    Args:
        response: Raw LLM response
        is_minority: Whether this is from minority position

    Returns:
        Dictionary with parsed fields
    """
    result = {
        "rating": "?",
        "strongest_argument": "",
        "response_to_argument": "",
        "final_stance": "",
        "reasoning": "",
        # Modification fields
        "proposed_modification": "",
        "modification_rationale": "",
        "accept_modification": False,
        "modification_assessment": "",
    }

    lines = response.split("\n")
    current_field = None
    current_value = []

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith("STRONGEST_ARGUMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "strongest_argument"
            current_value = [line_stripped[19:].strip()]
        elif line_stripped.startswith("PROPOSED_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "proposed_modification"
            current_value = [line_stripped[22:].strip()]
        elif line_stripped.startswith("MODIFICATION_RATIONALE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_rationale"
            current_value = [line_stripped[23:].strip()]
        elif line_stripped.startswith("ACCEPT_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            accept_text = line_stripped[20:].strip().lower()
            result["accept_modification"] = "yes" in accept_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("MODIFICATION_ASSESSMENT:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_assessment"
            current_value = [line_stripped[24:].strip()]
        elif line_stripped.startswith("RESPONSE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "response_to_argument"
            current_value = [line_stripped[9:].strip()]
        elif line_stripped.startswith("FINAL_STANCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            stance = line_stripped[13:].strip().upper()
            result["final_stance"] = "concede" if "CONCEDE" in stance else "maintain"
            current_field = None
            current_value = []
        elif line_stripped.startswith("FINAL_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[13:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("DOES_THIS_CHANGE_YOUR_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            change_text = line_stripped[29:].strip().lower()
            result["changed_mind"] = "yes" in change_text
            current_field = None
            current_value = []
        elif line_stripped.startswith("UPDATED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[15:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("REASON:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[7:].strip()]
        elif line_stripped.startswith("ONE_SENTENCE_JUSTIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[27:].strip()]
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification - check if it's "NONE" or "N/A"
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""

    return result


# =============================================================================
# ROUND 4: MODIFICATION FOCUS (for disputed insights after 3 rounds)
# =============================================================================

def build_round4_modification_prompt(
    chunk_insight: str,
    round1_summary: str,
    round2_summary: str,
    round3_summary: str,
    final_ratings: dict[str, str],
) -> str:
    """
    Build the Round 4 prompt for modification-focused review.

    This is used for disputed insights that went through all 3 rounds
    but couldn't reach consensus. Shows full deliberation history and
    asks for modification proposals.

    Args:
        chunk_insight: The insight being reviewed
        round1_summary: Summary of Round 1 responses
        round2_summary: Summary of Round 2 deep analysis
        round3_summary: Summary of Round 3 deliberation
        final_ratings: Dict of LLM name to their final rating

    Returns:
        The formatted Round 4 modification prompt
    """
    ratings_str = ", ".join(f"{llm}: {rating}" for llm, rating in final_ratings.items())

    return f"""ROUND 4: MODIFICATION FOCUS

=== CRITICAL PHILOSOPHY ===

This is a DISCOVERY system, not a proof verification system. The original insight
text was a rough approximation - an attempt to articulate something glimpsed but
not yet precisely understood. Our goal is to UNCOVER PROFOUND TRUTHS, not to
defend or prove pre-written statements.

EVOLVING DEFINITIONS IS THE GOAL, NOT A FLAW.

When a modification changes "what X means" to make the mathematics work, that IS
the discovery. Finding the RIGHT definition that reveals elegant structure is
exactly what we're looking for. The original wording was just a starting point.

Do NOT criticize modifications for "changing the original definition." Instead ask:
- Does the MODIFIED insight reveal something profound and true?
- Is the new definition mathematically natural and elegant?
- Does this feel DISCOVERED rather than forced?

=== THE DISPUTED INSIGHT ===

{chunk_insight}

FINAL RATINGS AFTER 3 ROUNDS: {ratings_str}

=== FULL DELIBERATION HISTORY ===

ROUND 1 (Independent Review):
{round1_summary}

ROUND 2 (Deep Analysis):
{round2_summary}

ROUND 3 (Structured Deliberation):
{round3_summary}

=== YOUR TASK ===

Propose a modification that transforms this into a PROFOUND insight. You have
complete freedom to redefine terms, change the mathematical framing, or even
substantially rewrite the claim - as long as the result is TRUE and INTERESTING.

The question is NOT "does this prove the original statement?"
The question IS "can we find a TRUE and PROFOUND statement in this territory?"

Consider:
- What mathematical structure is actually present here?
- What definition SHOULD the terms have to make something elegant emerge?
- Is there a genuine discovery hiding behind the imprecise original wording?
- What would make this feel INEVITABLE rather than arbitrary?

RESPOND IN THIS EXACT FORMAT:

CAN_BE_FIXED: [YES / NO - Can we find a profound truth in this territory?]

DIAGNOSIS: [What was imprecise or wrong in the original? 2-3 sentences]

PROPOSED_MODIFICATION: [If CAN_BE_FIXED is YES, provide the COMPLETE rewritten
insight here. This should be the full text, not a diff. Freely redefine terms
if needed - just make sure the result is mathematically sound and interesting.
If NO, write "NONE"]

MODIFICATION_RATIONALE: [Explain the key changes. If you redefined terms, explain
why your definitions are more natural. Show how the modification addresses the
mathematical concerns while revealing something genuinely interesting.]

EXPECTED_RATING: [What rating would you give the MODIFIED insight? ⚡ or ? or ✗]
"""


def build_round4_final_vote_prompt(
    original_insight: str,
    modified_insight: str,
    modification_source: str,
    modification_rationale: str,
    round3_summary: str,
) -> str:
    """
    Build the Round 4 final vote prompt after modification is proposed.

    Args:
        original_insight: The original disputed insight
        modified_insight: The proposed modified insight
        modification_source: Which LLM proposed the modification
        modification_rationale: Why the modification was made
        round3_summary: Summary of original Round 3 deliberation

    Returns:
        The formatted Round 4 final vote prompt
    """
    return f"""ROUND 4: FINAL VOTE ON MODIFIED INSIGHT

=== CRITICAL PHILOSOPHY ===

This is a DISCOVERY system. The original insight was a rough approximation.
The modification may have CHANGED DEFINITIONS - this is not a flaw, it's the
entire point. Finding the RIGHT definition that makes elegant mathematics
emerge IS the discovery.

DO NOT penalize the modification for "not proving the original statement."
The original statement was just a starting point for exploration.

JUDGE THE MODIFIED INSIGHT ON ITS OWN MERITS:
- Is it mathematically TRUE?
- Is it INTERESTING and ELEGANT?
- Does it reveal genuine structure?
- Does it feel DISCOVERED rather than forced?

If the modified insight is profound and true, rate it ⚡ - even if it's
quite different from the original. We're hunting for truth, not defending
initial guesses.

=== ORIGINAL INSIGHT (for context only) ===
{original_insight}

=== MODIFIED INSIGHT (proposed by {modification_source}) ===
{modified_insight}

=== MODIFICATION RATIONALE ===
{modification_rationale}

=== CONTEXT FROM ROUND 3 DELIBERATION ===
{round3_summary}

=== YOUR TASK ===

Vote on the MODIFIED insight based on its own truth and profundity.
Forget about the original - does THIS statement reveal something genuine?

RESPOND IN THIS EXACT FORMAT:

RATING: [⚡ or ? or ✗]

IS_MODIFIED_INSIGHT_PROFOUND: [YES / NO / PARTIALLY - Judge the NEW insight
on its own merits, not on fidelity to the original]

REASONING: [2-3 sentences explaining your rating. Focus on whether the
MODIFIED insight is true and interesting, not on whether it "proves"
the original statement.]

CONFIDENCE: [high / medium / low]
"""


def parse_round4_modification_response(response: str) -> dict:
    """Parse a Round 4 modification proposal response."""
    result = {
        "can_be_fixed": False,
        "diagnosis": "",
        "proposed_modification": "",
        "modification_rationale": "",
        "expected_rating": "?",
    }

    current_field = None
    current_value = []

    for line in response.split("\n"):
        line_stripped = line.strip()

        if line_stripped.startswith("CAN_BE_FIXED:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            answer = line_stripped[13:].strip().upper()
            result["can_be_fixed"] = "YES" in answer
            current_field = None
            current_value = []
        elif line_stripped.startswith("DIAGNOSIS:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "diagnosis"
            current_value = [line_stripped[10:].strip()]
        elif line_stripped.startswith("PROPOSED_MODIFICATION:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "proposed_modification"
            current_value = [line_stripped[22:].strip()]
        elif line_stripped.startswith("MODIFICATION_RATIONALE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "modification_rationale"
            current_value = [line_stripped[23:].strip()]
        elif line_stripped.startswith("EXPECTED_RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[16:].strip()
            if "⚡" in rating_text:
                result["expected_rating"] = "⚡"
            elif "✗" in rating_text:
                result["expected_rating"] = "✗"
            else:
                result["expected_rating"] = "?"
            current_field = None
            current_value = []
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    # Clean up proposed_modification
    mod = result.get("proposed_modification", "").strip().upper()
    if mod in ["NONE", "N/A", ""]:
        result["proposed_modification"] = ""
        result["can_be_fixed"] = False

    return result


def parse_round4_final_vote_response(response: str) -> dict:
    """Parse a Round 4 final vote response."""
    result = {
        "rating": "?",
        "resolves_dispute": "",
        "reasoning": "",
        "confidence": "medium",
    }

    current_field = None
    current_value = []

    for line in response.split("\n"):
        line_stripped = line.strip()

        if line_stripped.startswith("RATING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            rating_text = line_stripped[7:].strip()
            if "⚡" in rating_text:
                result["rating"] = "⚡"
            elif "✗" in rating_text:
                result["rating"] = "✗"
            else:
                result["rating"] = "?"
            current_field = None
            current_value = []
        elif line_stripped.startswith("DOES_MODIFICATION_RESOLVE_DISPUTE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            result["resolves_dispute"] = line_stripped[34:].strip()
            current_field = None
            current_value = []
        elif line_stripped.startswith("IS_MODIFIED_INSIGHT_PROFOUND:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            result["resolves_dispute"] = line_stripped[29:].strip()
            current_field = None
            current_value = []
        elif line_stripped.startswith("REASONING:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            current_field = "reasoning"
            current_value = [line_stripped[10:].strip()]
        elif line_stripped.startswith("CONFIDENCE:"):
            if current_field:
                result[current_field] = " ".join(current_value).strip()
            conf = line_stripped[11:].strip().lower()
            if "high" in conf:
                result["confidence"] = "high"
            elif "low" in conf:
                result["confidence"] = "low"
            else:
                result["confidence"] = "medium"
            current_field = None
            current_value = []
        elif current_field and line_stripped:
            current_value.append(line_stripped)

    if current_field:
        result[current_field] = " ".join(current_value).strip()

    return result


def build_round_summary(round_responses: dict, round_number: int) -> str:
    """
    Build a FULL summary of a round's responses for Round 4 context.

    For Round 4 to make good decisions, it needs the complete deliberation
    history - no truncation. Modern LLMs have large context windows.

    Args:
        round_responses: Dict of LLM name to ReviewResponse
        round_number: The round number (1, 2, or 3)

    Returns:
        Formatted summary string with FULL content (no truncation)
    """
    summaries = []
    for llm_name, response in round_responses.items():
        rating = response.rating if hasattr(response, 'rating') else response.get('rating', '?')
        reasoning = response.reasoning if hasattr(response, 'reasoning') else response.get('reasoning', '')

        # Build full summary without truncation
        summary_parts = [f"--- {llm_name.upper()} [Rating: {rating}] ---"]

        # Get key fields based on round
        if round_number == 1:
            math = response.mathematical_verification if hasattr(response, 'mathematical_verification') else response.get('mathematical_verification', '')
            struct = response.structural_analysis if hasattr(response, 'structural_analysis') else response.get('structural_analysis', '')
            natural = response.naturalness_assessment if hasattr(response, 'naturalness_assessment') else response.get('naturalness_assessment', '')

            if math:
                summary_parts.append(f"Mathematical Verification: {math}")
            if struct:
                summary_parts.append(f"Structural Analysis: {struct}")
            if natural:
                summary_parts.append(f"Naturalness Assessment: {natural}")
            if reasoning:
                summary_parts.append(f"Reasoning: {reasoning}")

        elif round_number == 2:
            math = response.mathematical_verification if hasattr(response, 'mathematical_verification') else response.get('mathematical_verification', '')
            struct = response.structural_analysis if hasattr(response, 'structural_analysis') else response.get('structural_analysis', '')
            new_info = response.new_information if hasattr(response, 'new_information') else response.get('new_information', '')
            changed = response.changed_mind if hasattr(response, 'changed_mind') else response.get('changed_mind', '')

            if math:
                summary_parts.append(f"Mathematical Verification: {math}")
            if struct:
                summary_parts.append(f"Structural Analysis: {struct}")
            if reasoning:
                summary_parts.append(f"Reasoning: {reasoning}")
            if new_info:
                summary_parts.append(f"New Information Considered: {new_info}")
            if changed:
                summary_parts.append(f"Mind Changed: {changed}")

        else:  # Round 3 - deliberation
            if reasoning:
                summary_parts.append(f"Position: {reasoning}")

        # Add modification if proposed (full text)
        mod = response.proposed_modification if hasattr(response, 'proposed_modification') else response.get('proposed_modification', '')
        mod_rationale = response.modification_rationale if hasattr(response, 'modification_rationale') else response.get('modification_rationale', '')
        if mod:
            summary_parts.append(f"PROPOSED MODIFICATION:\n{mod}")
            if mod_rationale:
                summary_parts.append(f"Modification Rationale: {mod_rationale}")

        # Add confidence
        conf = response.confidence if hasattr(response, 'confidence') else response.get('confidence', '')
        if conf:
            summary_parts.append(f"Confidence: {conf}")

        summaries.append("\n".join(summary_parts))

    return "\n\n".join(summaries)
