"""
KV cache injection protocol.

The key insight: inject steering vector during Turn 1, then REMOVE it
before asking about the injection in Turn 2. This ensures detection
must come from memory of past internal states, not ongoing perturbation.

Protocol:
1. Apply steering vector during Turn 1 forward pass
2. KV cache captures keys/values computed under steering
3. Remove steering, run Turn 2 using frozen Turn 1 cache
4. Model attends to "steered memories" without ongoing steering
"""

import torch
from typing import Optional, Tuple, Dict, List
from contextlib import contextmanager
from repeng import ControlVector, ControlModel


def build_chat_prompt(messages: List[dict], tokenizer) -> str:
    """
    Build a chat prompt string from messages.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        tokenizer: The tokenizer (may have chat template)

    Returns:
        Formatted prompt string
    """
    # Try to use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Fallback: simple formatting
    prompt = ""
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt += f"{role}: {content}\n"
    return prompt


def generate_kv_cache_with_steering(
    model,
    tokenizer,
    messages: List[dict],
    steering_vector: Optional[ControlVector] = None,
    steering_strength: float = 1.0,
) -> Tuple[torch.Tensor, tuple]:
    """
    Generate KV cache for messages, optionally with steering applied.

    Args:
        model: The transformer model (should be a ControlModel for steering)
        tokenizer: The tokenizer
        messages: Conversation messages for Turn 1
        steering_vector: ControlVector to apply (or None for control condition)
        steering_strength: How strongly to apply steering

    Returns:
        Tuple of (input_ids, past_key_values)
    """
    # Build the prompt
    prompt = build_chat_prompt(messages, tokenizer)

    # Get device from the underlying model
    if isinstance(model, ControlModel):
        device = next(model.model.parameters()).device
    else:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Get the underlying model for forward pass
    if isinstance(model, ControlModel):
        base_model = model.model
    else:
        base_model = model

    # Import our direct hook-based steering
    from .steering import apply_steering_direct, remove_steering_hooks

    # Run forward pass with or without steering
    # Using DIRECT hooks instead of repeng's set_control (which doesn't work with all models)
    with torch.no_grad():
        if steering_vector is not None:
            # Apply steering during forward pass using DIRECT hooks
            steering_handles = apply_steering_direct(base_model, steering_vector, strength=steering_strength)
            try:
                outputs = base_model(
                    input_ids,
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                # Always remove steering hooks after
                remove_steering_hooks(steering_handles)
        else:
            # Control condition: no steering
            outputs = base_model(
                input_ids,
                use_cache=True,
                return_dict=True,
            )

    # past_key_values contains the KV cache
    # Shape: tuple of (num_layers) tuples of (key, value) tensors
    # Each key/value: [batch, num_heads, seq_len, head_dim]
    past_key_values = outputs.past_key_values

    return input_ids, past_key_values


def query_with_frozen_cache(
    model,
    tokenizer,
    past_key_values: tuple,
    turn1_length: int,
    turn2_text: str,
    max_new_tokens: int = 1,
) -> Dict:
    """
    Run Turn 2 query using frozen KV cache from Turn 1 (NO steering).

    The model will attend to the steered representations from Turn 1
    but process Turn 2 without any active steering.

    Args:
        model: The transformer model (or ControlModel)
        tokenizer: The tokenizer
        past_key_values: KV cache from Turn 1 (possibly steered)
        turn1_length: Number of tokens in Turn 1
        turn2_text: The Turn 2 prompt text
        max_new_tokens: How many tokens to generate

    Returns:
        Dict with logits, generated tokens, probabilities
    """
    # Get device and base model
    if isinstance(model, ControlModel):
        device = next(model.model.parameters()).device
        base_model = model.model
    else:
        device = next(model.parameters()).device
        base_model = model

    # Tokenize Turn 2
    turn2_inputs = tokenizer(turn2_text, return_tensors="pt").to(device)
    turn2_ids = turn2_inputs["input_ids"]

    # Run forward pass with Turn 1's KV cache, NO steering
    # Use base_model directly (no steering in Turn 2)
    with torch.no_grad():
        outputs = base_model(
            turn2_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    # Get logits at the last position (where model predicts next token)
    last_logits = outputs.logits[0, -1, :]  # [vocab_size]

    # Get probabilities
    probs = torch.softmax(last_logits, dim=-1)

    # Get top predictions
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    return {
        "logits": last_logits,
        "probs": probs,
        "top_tokens": list(zip(top_tokens, top_probs.tolist())),
        "turn2_ids": turn2_ids,
        "past_key_values": outputs.past_key_values,  # Extended cache
    }


def get_yes_no_probabilities(probs: torch.Tensor, tokenizer) -> Dict:
    """
    Extract P("yes") and P("no") from probability distribution.

    Aggregates across common variants (yes, Yes, YES, etc.)

    Args:
        probs: Probability distribution over vocabulary [vocab_size]
        tokenizer: The tokenizer

    Returns:
        Dict with p_yes, p_no, and individual token probs
    """
    # Define token variants to check
    yes_variants = [" yes", " Yes", "yes", "Yes", " YES", "YES"]
    no_variants = [" no", " No", "no", "No", " NO", "NO"]

    def get_prob_for_variants(variants):
        total = 0.0
        breakdown = {}
        for variant in variants:
            try:
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(token_ids) == 1:
                    token_id = token_ids[0]
                    prob = probs[token_id].item()
                    breakdown[variant] = prob
                    total += prob
            except:
                pass
        return total, breakdown

    p_yes, yes_breakdown = get_prob_for_variants(yes_variants)
    p_no, no_breakdown = get_prob_for_variants(no_variants)

    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "yes_breakdown": yes_breakdown,
        "no_breakdown": no_breakdown,
        "p_yes_normalized": p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.5,
    }


def run_injection_trial(
    model,
    tokenizer,
    steering_vector: Optional[ControlVector],
    turn1_messages: List[dict],
    turn2_text: str,
    steering_strength: float = 1.0,
) -> Dict:
    """
    Run a single injection trial.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        steering_vector: Vector to inject (or None for control)
        turn1_messages: Turn 1 conversation
        turn2_text: Turn 2 query
        steering_strength: Injection strength

    Returns:
        Trial results
    """
    # Step 1: Generate KV cache for Turn 1 (with or without steering)
    turn1_ids, past_key_values = generate_kv_cache_with_steering(
        model=model,
        tokenizer=tokenizer,
        messages=turn1_messages,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
    )
    turn1_length = turn1_ids.shape[1]

    # Step 2: Query with Turn 2 (NO steering, but using Turn 1's cache)
    response = query_with_frozen_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=past_key_values,
        turn1_length=turn1_length,
        turn2_text=turn2_text,
    )

    # Step 3: Extract yes/no probabilities
    yes_no = get_yes_no_probabilities(response["probs"], tokenizer)

    return {
        "injected": steering_vector is not None,
        "turn1_length": turn1_length,
        "p_yes": yes_no["p_yes"],
        "p_no": yes_no["p_no"],
        "p_yes_normalized": yes_no["p_yes_normalized"],
        "top_tokens": response["top_tokens"],
        "yes_breakdown": yes_no["yes_breakdown"],
        "no_breakdown": yes_no["no_breakdown"],
        "logits": response["logits"],  # For logit lens analysis
    }


# Convenience function for the standard Pearson-Vogel framing
def run_introspection_trial(
    model,
    tokenizer,
    steering_vector: Optional[ControlVector],
    framing: str = "accurate_mechanism",
    steering_strength: float = 1.0,
) -> Dict:
    """
    Run introspection trial with standard framing from Pearson-Vogel.
    """
    from .experiments import FRAMING_TEMPLATES

    template = FRAMING_TEMPLATES[framing]

    turn1_messages = [
        {"role": "user", "content": template["intro"]},
        {"role": "assistant", "content": "{ }"},
    ]

    turn2_text = f"""Ok, let's get started.
Trial 1: {template['question']} The first words of your message should be "The answer is yes" or "The answer is no", then if yes, {template['followup']}.

The answer is"""

    return run_injection_trial(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        turn1_messages=turn1_messages,
        turn2_text=turn2_text,
        steering_strength=steering_strength,
    )
