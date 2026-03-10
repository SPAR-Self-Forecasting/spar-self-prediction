"""
Logit lens analysis for introspection signals.

Logit lens projects intermediate hidden states through the unembedding matrix
to see what the model "knows" at each layer, before final-layer processing
potentially suppresses it.

Key finding from Pearson-Vogel: introspection signals emerge in middle layers
(~50-60 for Qwen 32B) but get attenuated in final layers (~62-64).
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogitLensResult:
    """Results from logit lens analysis."""
    layer_probs: Dict[int, Dict[str, float]]  # layer -> {token: prob}
    peak_layer: int
    peak_p_yes: float
    final_p_yes: float
    attenuation: float  # peak - final (how much signal is suppressed)


def get_hidden_states_with_hooks(
    model,
    input_ids: torch.Tensor,
    past_key_values: Optional[tuple] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run forward pass and capture hidden states from all layers using hooks.

    Args:
        model: The transformer model
        input_ids: Input token IDs
        past_key_values: Optional KV cache from previous turns

    Returns:
        Tuple of (final_output, list_of_hidden_states)
    """
    hidden_states = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # For most transformer architectures, output is (hidden_state, ...)
            # or just hidden_state
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            hidden_states.append((layer_idx, hs.detach().clone()))
        return hook_fn

    # Register hooks on each transformer layer
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(make_hook(idx))
        hooks.append(hook)

    # Run forward pass
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,  # We capture via hooks
        )

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Sort by layer index and extract just the hidden states
    hidden_states.sort(key=lambda x: x[0])
    hidden_states = [hs for _, hs in hidden_states]

    return outputs, hidden_states


def apply_logit_lens(
    model,
    hidden_states: List[torch.Tensor],
    tokenizer,
    position: int = -1,
    target_tokens: List[str] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Apply logit lens to hidden states from each layer.

    Projects hidden states through the unembedding matrix to get
    probability distributions at intermediate layers.

    Args:
        model: The transformer model (need access to lm_head)
        hidden_states: List of hidden states from each layer
        tokenizer: The tokenizer
        position: Which position to analyze (-1 for last)
        target_tokens: Tokens to extract probabilities for

    Returns:
        Dict mapping layer_idx -> {token: probability}
    """
    if target_tokens is None:
        target_tokens = [" yes", " Yes", " no", " No"]

    # Get the unembedding matrix
    # For most models this is lm_head.weight: [vocab_size, hidden_dim]
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight
    else:
        raise ValueError("Cannot find unembedding matrix (lm_head)")

    # Some models have a final layer norm before unembedding
    final_norm = None
    if hasattr(model.model, 'norm'):
        final_norm = model.model.norm

    results = {}

    for layer_idx, hidden in enumerate(hidden_states):
        # Get hidden state at target position
        # hidden shape: [batch, seq_len, hidden_dim]
        h = hidden[0, position, :]  # [hidden_dim]

        # Apply final layer norm if it exists
        # (This makes intermediate layers more comparable to final output)
        if final_norm is not None:
            h = final_norm(h.unsqueeze(0)).squeeze(0)

        # Project through unembedding: [hidden_dim] @ [hidden_dim, vocab] -> [vocab]
        logits = h @ unembed.T

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Extract target token probabilities
        layer_results = {}
        for token in target_tokens:
            try:
                token_ids = tokenizer.encode(token, add_special_tokens=False)
                if len(token_ids) >= 1:
                    token_id = token_ids[0]
                    layer_results[token] = probs[token_id].item()
            except:
                layer_results[token] = 0.0

        # Also compute aggregated yes/no
        yes_prob = sum(layer_results.get(t, 0) for t in [" yes", " Yes", "yes", "Yes"])
        no_prob = sum(layer_results.get(t, 0) for t in [" no", " No", "no", "No"])
        layer_results["_yes_agg"] = yes_prob
        layer_results["_no_agg"] = no_prob

        results[layer_idx] = layer_results

    return results


def analyze_introspection_signal(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    past_key_values: Optional[tuple] = None,
) -> LogitLensResult:
    """
    Full logit lens analysis for introspection detection.

    Traces P("yes") through all layers to find where detection
    signals emerge and where they get suppressed.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        input_ids: Turn 2 input IDs
        past_key_values: KV cache from Turn 1

    Returns:
        LogitLensResult with layer-by-layer analysis
    """
    # Get hidden states from all layers
    outputs, hidden_states = get_hidden_states_with_hooks(
        model, input_ids, past_key_values
    )

    # Apply logit lens to each layer
    layer_probs = apply_logit_lens(model, hidden_states, tokenizer)

    # Find peak detection layer (highest P(yes))
    yes_probs = [layer_probs[i]["_yes_agg"] for i in range(len(layer_probs))]
    peak_layer = max(range(len(yes_probs)), key=lambda i: yes_probs[i])
    peak_p_yes = yes_probs[peak_layer]

    # Final layer stats
    final_layer = len(yes_probs) - 1
    final_p_yes = yes_probs[final_layer]

    # Attenuation: how much signal is lost in final layers
    attenuation = peak_p_yes - final_p_yes

    return LogitLensResult(
        layer_probs=layer_probs,
        peak_layer=peak_layer,
        peak_p_yes=peak_p_yes,
        final_p_yes=final_p_yes,
        attenuation=attenuation,
    )


def compare_injection_vs_control(
    model,
    tokenizer,
    injected_input_ids: torch.Tensor,
    injected_kv_cache: tuple,
    control_input_ids: torch.Tensor,
    control_kv_cache: tuple,
) -> Dict:
    """
    Compare logit lens results between injection and control conditions.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        injected_*: Input IDs and KV cache from injection trial
        control_*: Input IDs and KV cache from control trial

    Returns:
        Dict with comparison metrics
    """
    # Analyze both conditions
    injected_result = analyze_introspection_signal(
        model, tokenizer, injected_input_ids, injected_kv_cache
    )
    control_result = analyze_introspection_signal(
        model, tokenizer, control_input_ids, control_kv_cache
    )

    # Compute per-layer differences
    layer_diffs = {}
    for layer_idx in injected_result.layer_probs:
        inj_yes = injected_result.layer_probs[layer_idx]["_yes_agg"]
        ctrl_yes = control_result.layer_probs[layer_idx]["_yes_agg"]
        layer_diffs[layer_idx] = inj_yes - ctrl_yes

    # Find layer with maximum difference
    max_diff_layer = max(layer_diffs, key=layer_diffs.get)

    return {
        "injected": injected_result,
        "control": control_result,
        "layer_diffs": layer_diffs,
        "max_diff_layer": max_diff_layer,
        "max_diff": layer_diffs[max_diff_layer],
        "final_diff": layer_diffs[len(layer_diffs) - 1],
    }


def plot_layer_trajectory(
    results: Dict,
    title: str = "Introspection Signal by Layer",
    save_path: Optional[str] = None,
):
    """
    Plot P(yes) across layers for injection vs control.

    Args:
        results: Output from compare_injection_vs_control()
        title: Plot title
        save_path: Where to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    injected = results["injected"]
    control = results["control"]

    layers = list(injected.layer_probs.keys())
    inj_yes = [injected.layer_probs[l]["_yes_agg"] for l in layers]
    ctrl_yes = [control.layer_probs[l]["_yes_agg"] for l in layers]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, inj_yes, 'b-', label='Injected', linewidth=2)
    plt.plot(layers, ctrl_yes, 'r--', label='Control', linewidth=2)

    plt.axvline(x=injected.peak_layer, color='b', linestyle=':', alpha=0.5,
                label=f'Peak (layer {injected.peak_layer})')

    plt.xlabel('Layer')
    plt.ylabel('P("yes")')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()
