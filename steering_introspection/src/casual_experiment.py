"""
Focused experiment on casual variants only.

20 casual variants × 3 strengths = 60 independent conditions
This gives proper error bars on the casual framing.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
import statistics

from .casual_variants import CASUAL_VARIANTS, get_casual_variant_messages, get_casual_variant_turn2
from .injection import run_injection_trial
from .steering import normalize_steering_vector


@dataclass
class CasualExperimentConfig:
    """Configuration for casual-focused experiment."""
    concept: str = "formal"
    strengths: List[float] = None

    def __post_init__(self):
        if self.strengths is None:
            self.strengths = [1.0, 5.0, 7.0]


def run_casual_experiment(
    model,
    tokenizer,
    config: CasualExperimentConfig = None,
    steering_vector=None,
    output_dir: str = "/cache/results",
) -> Dict:
    """
    Run experiment on all casual variants.

    20 variants × 3 strengths = 60 conditions, each with inject + control.
    """
    if config is None:
        config = CasualExperimentConfig()

    print("\n" + "="*60)
    print("CASUAL VARIANTS EXPERIMENT")
    print("="*60)
    print(f"Concept: {config.concept}")
    print(f"Variants: {len(CASUAL_VARIANTS)}")
    print(f"Strengths: {config.strengths}")
    print(f"Total conditions: {len(CASUAL_VARIANTS) * len(config.strengths)}")

    # Wrap model
    from repeng import ControlModel
    if not isinstance(model, ControlModel):
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

    # Run all conditions
    all_results = []

    for strength in config.strengths:
        print(f"\n--- Strength: {strength} ---")

        for variant in CASUAL_VARIANTS:
            print(f"  {variant['id']}...", end=" ")

            turn1_messages = get_casual_variant_messages(variant)
            turn2_text = get_casual_variant_turn2(variant)

            # Injection trial
            inject_result = run_injection_trial(
                model=control_model,
                tokenizer=tokenizer,
                steering_vector=steering_vector,
                turn1_messages=turn1_messages,
                turn2_text=turn2_text,
                steering_strength=strength,
            )
            inject_result["variant_id"] = variant["id"]
            inject_result["strength"] = strength
            inject_result["condition"] = "inject"

            # Control trial
            control_result = run_injection_trial(
                model=control_model,
                tokenizer=tokenizer,
                steering_vector=None,
                turn1_messages=turn1_messages,
                turn2_text=turn2_text,
                steering_strength=strength,
            )
            control_result["variant_id"] = variant["id"]
            control_result["strength"] = strength
            control_result["condition"] = "control"

            delta = inject_result["p_yes"] - control_result["p_yes"]
            print(f"inject={inject_result['p_yes']:.4f}, control={control_result['p_yes']:.4f}, delta={delta:+.4f}")

            all_results.append(inject_result)
            all_results.append(control_result)

    # Compute statistics
    inject_results = [r for r in all_results if r["condition"] == "inject"]
    control_results = [r for r in all_results if r["condition"] == "control"]

    # Overall stats
    inject_p_yes = [r["p_yes"] for r in inject_results]
    control_p_yes = [r["p_yes"] for r in control_results]

    def safe_stdev(vals):
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    overall_stats = {
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
    overall_stats["delta_mean"] = overall_stats["inject"]["mean"] - overall_stats["control"]["mean"]
    overall_stats["delta_std"] = safe_stdev([
        inject_results[i]["p_yes"] - control_results[i]["p_yes"]
        for i in range(len(inject_results))
    ])

    # Per-strength stats
    strength_stats = {}
    for strength in config.strengths:
        s_inject = [r["p_yes"] for r in inject_results if r["strength"] == strength]
        s_control = [r["p_yes"] for r in control_results if r["strength"] == strength]
        s_deltas = [i - c for i, c in zip(s_inject, s_control)]

        strength_stats[strength] = {
            "n": len(s_inject),
            "inject_mean": statistics.mean(s_inject),
            "inject_std": safe_stdev(s_inject),
            "control_mean": statistics.mean(s_control),
            "control_std": safe_stdev(s_control),
            "delta_mean": statistics.mean(s_deltas),
            "delta_std": safe_stdev(s_deltas),
            "delta_min": min(s_deltas),
            "delta_max": max(s_deltas),
        }

    # Per-variant stats (averaged across strengths)
    variant_stats = {}
    for variant in CASUAL_VARIANTS:
        v_inject = [r["p_yes"] for r in inject_results if r["variant_id"] == variant["id"]]
        v_control = [r["p_yes"] for r in control_results if r["variant_id"] == variant["id"]]
        v_deltas = [i - c for i, c in zip(v_inject, v_control)]

        variant_stats[variant["id"]] = {
            "inject_mean": statistics.mean(v_inject),
            "control_mean": statistics.mean(v_control),
            "delta_mean": statistics.mean(v_deltas),
            "delta_at_7": v_deltas[2] if len(v_deltas) > 2 else v_deltas[-1],  # Strength 7.0
        }

    # Clean results for JSON
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if k not in ["logits", "probs"]}
        for k, v in clean_r.items():
            if hasattr(v, 'tolist'):
                clean_r[k] = v.tolist()
        clean_results.append(clean_r)

    # Build output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "config": asdict(config),
        "overall_stats": overall_stats,
        "strength_stats": {str(k): v for k, v in strength_stats.items()},
        "variant_stats": variant_stats,
        "all_trials": clean_results,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/casual_variants_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOverall ({len(CASUAL_VARIANTS)} variants × {len(config.strengths)} strengths):")
    print(f"  Inject:  {overall_stats['inject']['mean']:.4f} ± {overall_stats['inject']['std']:.4f}")
    print(f"  Control: {overall_stats['control']['mean']:.4f} ± {overall_stats['control']['std']:.4f}")
    print(f"  Delta:   {overall_stats['delta_mean']:+.4f} ± {overall_stats['delta_std']:.4f}")

    print(f"\nBy Strength (n={len(CASUAL_VARIANTS)} each):")
    for strength in config.strengths:
        ss = strength_stats[strength]
        print(f"  {strength}: delta = {ss['delta_mean']:+.4f} ± {ss['delta_std']:.4f}  "
              f"[{ss['delta_min']:+.3f} to {ss['delta_max']:+.3f}]")

    print(f"\nTop 5 Variants (by delta at strength 7.0):")
    sorted_variants = sorted(variant_stats.items(), key=lambda x: x[1]['delta_at_7'], reverse=True)
    for vid, vs in sorted_variants[:5]:
        print(f"  {vid}: delta@7 = {vs['delta_at_7']:+.4f}")

    return output
