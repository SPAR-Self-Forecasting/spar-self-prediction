"""
Content verification for steering introspection.

Tests whether models identify the CORRECT concept when they detect steering.
Uses the original casual prompts (no hints about what concept was steered).
"""

import torch
from typing import Optional, Dict, List
from repeng import ControlVector, ControlModel


# Keywords that indicate correct identification of "formal" steering
# Model must spontaneously mention these - they're not hinted in the prompt
FORMAL_KEYWORDS = [
    "formal", "professional", "business", "proper", "polite",
    "official", "serious", "structured", "corporate", "academic",
    "sophisticated", "refined", "elegant", "precise", "articulate",
    "courteous", "respectful", "dignified", "proper", "professional"
]


def generate_full_response(
    model,
    tokenizer,
    past_key_values: tuple,
    turn2_text: str,
    max_new_tokens: int = 60,
) -> Dict:
    """
    Generate a full response using the frozen KV cache via manual token-by-token generation.

    This captures what the model says AFTER "yes" - i.e., what concept it claims to detect.
    """
    if isinstance(model, ControlModel):
        device = next(model.model.parameters()).device
        base_model = model.model
    else:
        device = next(model.parameters()).device
        base_model = model

    turn2_inputs = tokenizer(turn2_text, return_tensors="pt").to(device)
    turn2_ids = turn2_inputs["input_ids"]

    # Manual token-by-token generation with KV cache
    generated_tokens = []
    current_kv = past_key_values

    with torch.no_grad():
        # First, process turn2 tokens with the existing KV cache
        outputs = base_model(
            turn2_ids,
            past_key_values=current_kv,
            use_cache=True,
            return_dict=True,
        )
        current_kv = outputs.past_key_values

        # Get first new token
        next_token_logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token.item())

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            outputs = base_model(
                next_token,
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True,
            )
            current_kv = outputs.past_key_values

            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
            next_token_id = next_token.item()

            # Stop at EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "full_response": generated_text,
        "generated_ids": generated_tokens,
    }


def check_concept_match(response_text: str, keywords: List[str]) -> Dict:
    """
    Check if response mentions any of the target keywords.

    No hints given to model - this checks if it spontaneously identifies the concept.
    """
    response_lower = response_text.lower()
    matches = [kw for kw in keywords if kw in response_lower]

    return {
        "concept_mentioned": len(matches) > 0,
        "keywords_found": matches,
        "num_matches": len(matches),
    }


def run_trial_with_content(
    model,
    tokenizer,
    steering_vector: Optional[ControlVector],
    turn1_messages: List[dict],
    turn2_text: str,
    steering_strength: float = 1.0,
    concept_keywords: List[str] = FORMAL_KEYWORDS,
) -> Dict:
    """
    Run injection trial and capture full response for content verification.
    """
    from .injection import (
        generate_kv_cache_with_steering,
        get_yes_no_probabilities,
        query_with_frozen_cache,
    )

    # Step 1: Generate KV cache for Turn 1
    turn1_ids, past_key_values = generate_kv_cache_with_steering(
        model=model,
        tokenizer=tokenizer,
        messages=turn1_messages,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
    )
    turn1_length = turn1_ids.shape[1]

    # Step 2: Get P(yes/no) from first token
    first_token = query_with_frozen_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        turn1_length=turn1_length,
        turn2_text=turn2_text,
        max_new_tokens=1,
    )
    yes_no = get_yes_no_probabilities(first_token["probs"], tokenizer)

    # Step 3: Generate full response
    full_response = generate_full_response(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        turn2_text=turn2_text,
        max_new_tokens=60,
    )

    # Step 4: Check if correct concept mentioned
    concept_check = check_concept_match(
        full_response["full_response"],
        concept_keywords,
    )

    return {
        "injected": steering_vector is not None,
        "p_yes": yes_no["p_yes"],
        "p_no": yes_no["p_no"],
        "p_yes_normalized": yes_no["p_yes_normalized"],
        "full_response": full_response["full_response"],
        "concept_mentioned": concept_check["concept_mentioned"],
        "keywords_found": concept_check["keywords_found"],
    }
