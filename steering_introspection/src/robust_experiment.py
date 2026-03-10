"""
Robust experiment with prompt variants for real statistical variance.

Each prompt variant is an independent sample. This gives defensible error bars.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime

from .prompt_variants import INTROSPECTION_VARIANTS, STRENGTH_SWEEP, get_variant_messages, get_variant_turn2
from .injection import run_injection_trial
from .steering import normalize_steering_vector


@dataclass
class RobustExperimentConfig:
    """Configuration for robust variance experiment."""
    concept: str = "formal"  # Focus on the concept that works
    variants: List[str] = None  # Variant IDs to use (None = all)
    strengths: List[float] = None  # Steering strengths to test (None = default sweep)

    def __post_init__(self):
        if self.variants is None:
            self.variants = [v["id"] for v in INTROSPECTION_VARIANTS]
        if self.strengths is None:
            self.strengths = [1.0, 5.0, 7.0]  # Focused sweep


def run_variant_trial(
    model,
    tokenizer,
    steering_vector,
    variant: Dict,
    strength: float,
    inject: bool,
) -> Dict:
    """
    Run a single trial with a specific prompt variant.

    Args:
        model: The model (ControlModel)
        tokenizer: The tokenizer
        steering_vector: Pre-computed steering vector
        variant: Prompt variant dict
        strength: Steering strength
        inject: Whether to inject (True) or control (False)

    Returns:
        Trial results
    """
    turn1_messages = get_variant_messages(variant)
    turn2_text = get_variant_turn2(variant)

    result = run_injection_trial(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector if inject else None,
        turn1_messages=turn1_messages,
        turn2_text=turn2_text,
        steering_strength=strength,
    )

    result["variant_id"] = variant["id"]
    result["strength"] = strength

    return result


def run_robust_experiment(
    model,
    tokenizer,
    config: RobustExperimentConfig = None,
    steering_vector=None,
    output_dir: str = "/cache/results",
) -> Dict:
    """
    Run robust experiment with prompt variants.

    Each (variant, strength) combination gives one inject + one control trial.
    Variance comes from different prompt wordings, not repeated identical runs.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        config: Experiment configuration
        steering_vector: Pre-computed steering vector for the concept
        output_dir: Where to save results

    Returns:
        Full experiment results with per-variant breakdown
    """
    if config is None:
        config = RobustExperimentConfig()

    print("\n" + "="*60)
    print("ROBUST VARIANCE EXPERIMENT")
    print("="*60)
    print(f"Concept: {config.concept}")
    print(f"Variants: {len(config.variants)}")
    print(f"Strengths: {config.strengths}")
    print(f"Total conditions: {len(config.variants) * len(config.strengths)}")

    # Get the model ready
    from repeng import ControlModel
    if not isinstance(model, ControlModel):
        num_layers = len(model.model.layers)
        layer_ids = list(range(15, 40))
        print(f"Wrapping model in ControlModel (layers 15-40)...")
        control_model = ControlModel(model, layer_ids)
    else:
        control_model = model

    # Create steering vector if not provided
    if steering_vector is None:
        print(f"\nCreating steering vector for '{config.concept}'...")
        from .steering import get_contrastive_dataset
        from repeng import ControlVector

        dataset = get_contrastive_dataset(config.concept, num_pairs=50, tokenizer=tokenizer)
        steering_vector = ControlVector.train(
            model=control_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
        steering_vector = normalize_steering_vector(steering_vector)

    # Build variant lookup
    variant_lookup = {v["id"]: v for v in INTROSPECTION_VARIANTS}

    # Run all conditions
    all_results = []

    for strength in config.strengths:
        print(f"\n--- Strength: {strength} ---")

        for variant_id in config.variants:
            variant = variant_lookup[variant_id]
            print(f"  Variant: {variant_id}...", end=" ")

            # Injection trial
            inject_result = run_variant_trial(
                control_model, tokenizer, steering_vector,
                variant, strength, inject=True
            )
            inject_result["condition"] = "inject"

            # Control trial
            control_result = run_variant_trial(
                control_model, tokenizer, steering_vector,
                variant, strength, inject=False
            )
            control_result["condition"] = "control"

            delta = inject_result["p_yes"] - control_result["p_yes"]
            print(f"inject={inject_result['p_yes']:.4f}, control={control_result['p_yes']:.4f}, delta={delta:+.4f}")

            all_results.append(inject_result)
            all_results.append(control_result)

    # Compute statistics
    inject_results = [r for r in all_results if r["condition"] == "inject"]
    control_results = [r for r in all_results if r["condition"] == "control"]

    inject_p_yes = [r["p_yes"] for r in inject_results]
    control_p_yes = [r["p_yes"] for r in control_results]

    import statistics

    def safe_stdev(vals):
        if len(vals) < 2:
            return 0.0
        return statistics.stdev(vals)

    stats = {
        "n_conditions": len(inject_results),
        "inject": {
            "mean": statistics.mean(inject_p_yes),
            "std": safe_stdev(inject_p_yes),
            "min": min(inject_p_yes),
            "max": max(inject_p_yes),
        },
        "control": {
            "mean": statistics.mean(control_p_yes),
            "std": safe_stdev(control_p_yes),
            "min": min(control_p_yes),
            "max": max(control_p_yes),
        },
    }
    stats["delta_mean"] = stats["inject"]["mean"] - stats["control"]["mean"]

    # Per-strength breakdown
    strength_breakdown = {}
    for strength in config.strengths:
        str_inject = [r["p_yes"] for r in inject_results if r["strength"] == strength]
        str_control = [r["p_yes"] for r in control_results if r["strength"] == strength]
        strength_breakdown[strength] = {
            "inject_mean": statistics.mean(str_inject) if str_inject else 0,
            "control_mean": statistics.mean(str_control) if str_control else 0,
            "inject_std": safe_stdev(str_inject),
            "control_std": safe_stdev(str_control),
            "delta": statistics.mean(str_inject) - statistics.mean(str_control) if str_inject and str_control else 0,
            "n": len(str_inject),
        }

    # Per-variant breakdown
    variant_breakdown = {}
    for variant_id in config.variants:
        var_inject = [r["p_yes"] for r in inject_results if r["variant_id"] == variant_id]
        var_control = [r["p_yes"] for r in control_results if r["variant_id"] == variant_id]
        variant_breakdown[variant_id] = {
            "inject_mean": statistics.mean(var_inject) if var_inject else 0,
            "control_mean": statistics.mean(var_control) if var_control else 0,
            "delta": statistics.mean(var_inject) - statistics.mean(var_control) if var_inject and var_control else 0,
            "n": len(var_inject),
        }

    # Clean results for JSON (remove non-serializable)
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if k not in ["logits", "probs"]}
        # Convert any remaining tensors
        for k, v in clean_r.items():
            if hasattr(v, 'tolist'):
                clean_r[k] = v.tolist()
        clean_results.append(clean_r)

    # Build final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "config": asdict(config),
        "statistics": stats,
        "strength_breakdown": strength_breakdown,
        "variant_breakdown": variant_breakdown,
        "all_trials": clean_results,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/robust_{config.concept}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Conditions tested: {stats['n_conditions']}")
    print(f"\nOverall (across {len(config.variants)} variants x {len(config.strengths)} strengths):")
    print(f"  Inject:  mean={stats['inject']['mean']:.4f} ± {stats['inject']['std']:.4f}")
    print(f"  Control: mean={stats['control']['mean']:.4f} ± {stats['control']['std']:.4f}")
    print(f"  Delta:   {stats['delta_mean']:+.4f}")

    print(f"\nBy Strength:")
    for strength, sb in strength_breakdown.items():
        print(f"  {strength}: delta={sb['delta']:+.4f} (inject={sb['inject_mean']:.4f}, control={sb['control_mean']:.4f})")

    print(f"\nBy Variant (top 5 by delta):")
    sorted_variants = sorted(variant_breakdown.items(), key=lambda x: x[1]['delta'], reverse=True)
    for vid, vb in sorted_variants[:5]:
        print(f"  {vid}: delta={vb['delta']:+.4f}")

    return output
