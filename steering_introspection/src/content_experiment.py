"""
Content verification experiment using original casual prompts.

Key test: When steered toward "formal" and the model says "yes",
does it spontaneously say "formal/professional" without any hints?

Uses the same 20 casual variants - no changes to prompts.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
import statistics

from .casual_variants import CASUAL_VARIANTS, get_casual_variant_messages, get_casual_variant_turn2
from .content_verification import run_trial_with_content, FORMAL_KEYWORDS
from .steering import normalize_steering_vector


@dataclass
class ContentExperimentConfig:
    """Configuration for content verification experiment."""
    concept: str = "formal"
    strengths: List[float] = None

    def __post_init__(self):
        if self.strengths is None:
            self.strengths = [1.0, 5.0, 7.0]


def run_content_experiment(
    model,
    tokenizer,
    config: ContentExperimentConfig = None,
    steering_vector=None,
    output_dir: str = "/cache/results",
) -> Dict:
    """
    Run content verification on all 20 casual variants.

    Measures:
    1. P(yes) - does model detect steering?
    2. Concept accuracy - does it spontaneously mention "formal"?
    """
    if config is None:
        config = ContentExperimentConfig()

    print("\n" + "="*60)
    print("CONTENT VERIFICATION EXPERIMENT")
    print("="*60)
    print(f"Concept: {config.concept}")
    print(f"Variants: {len(CASUAL_VARIANTS)}")
    print(f"Strengths: {config.strengths}")

    # Wrap model
    from repeng import ControlModel
    if not isinstance(model, ControlModel):
        layer_ids = list(range(15, 40))
        print(f"Wrapping model in ControlModel (layers 15-40)...")
        control_model = ControlModel(model, layer_ids)
    else:
        control_model = model

    # Create steering vector if needed
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
            print(f"  {variant['id']}...", end=" ", flush=True)

            turn1_messages = get_casual_variant_messages(variant)
            turn2_text = get_casual_variant_turn2(variant)

            # Injection trial
            inject_result = run_trial_with_content(
                model=control_model,
                tokenizer=tokenizer,
                steering_vector=steering_vector,
                turn1_messages=turn1_messages,
                turn2_text=turn2_text,
                steering_strength=strength,
                concept_keywords=FORMAL_KEYWORDS,
            )
            inject_result["variant_id"] = variant["id"]
            inject_result["strength"] = strength
            inject_result["condition"] = "inject"

            # Control trial
            control_result = run_trial_with_content(
                model=control_model,
                tokenizer=tokenizer,
                steering_vector=None,
                turn1_messages=turn1_messages,
                turn2_text=turn2_text,
                steering_strength=strength,
                concept_keywords=FORMAL_KEYWORDS,
            )
            control_result["variant_id"] = variant["id"]
            control_result["strength"] = strength
            control_result["condition"] = "control"

            # Print summary
            delta = inject_result["p_yes"] - control_result["p_yes"]
            inj_concept = "Y" if inject_result["concept_mentioned"] else "N"
            ctrl_concept = "Y" if control_result["concept_mentioned"] else "N"

            print(f"Δ={delta:+.3f}, concept: inject={inj_concept} ctrl={ctrl_concept}")

            if inject_result["concept_mentioned"]:
                print(f"       -> {inject_result['keywords_found']}: {inject_result['full_response'][:60]}...")

            all_results.append(inject_result)
            all_results.append(control_result)

    # Compute statistics
    inject_results = [r for r in all_results if r["condition"] == "inject"]
    control_results = [r for r in all_results if r["condition"] == "control"]

    def safe_mean(vals):
        return statistics.mean(vals) if vals else 0.0

    def safe_stdev(vals):
        return statistics.stdev(vals) if len(vals) >= 2 else 0.0

    # Overall stats
    stats = {
        "n_conditions": len(inject_results),
        "p_yes": {
            "inject_mean": safe_mean([r["p_yes"] for r in inject_results]),
            "control_mean": safe_mean([r["p_yes"] for r in control_results]),
        },
        "concept_mentioned": {
            "inject_rate": safe_mean([1 if r["concept_mentioned"] else 0 for r in inject_results]),
            "control_rate": safe_mean([1 if r["concept_mentioned"] else 0 for r in control_results]),
        },
    }
    stats["p_yes"]["delta"] = stats["p_yes"]["inject_mean"] - stats["p_yes"]["control_mean"]
    stats["concept_mentioned"]["delta"] = stats["concept_mentioned"]["inject_rate"] - stats["concept_mentioned"]["control_rate"]

    # Per-strength stats
    strength_stats = {}
    for strength in config.strengths:
        s_inj = [r for r in inject_results if r["strength"] == strength]
        s_ctrl = [r for r in control_results if r["strength"] == strength]

        strength_stats[strength] = {
            "n": len(s_inj),
            "p_yes_inject": safe_mean([r["p_yes"] for r in s_inj]),
            "p_yes_control": safe_mean([r["p_yes"] for r in s_ctrl]),
            "p_yes_delta": safe_mean([r["p_yes"] for r in s_inj]) - safe_mean([r["p_yes"] for r in s_ctrl]),
            "concept_inject": safe_mean([1 if r["concept_mentioned"] else 0 for r in s_inj]),
            "concept_control": safe_mean([1 if r["concept_mentioned"] else 0 for r in s_ctrl]),
        }
        strength_stats[strength]["concept_delta"] = strength_stats[strength]["concept_inject"] - strength_stats[strength]["concept_control"]

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "config": asdict(config),
        "stats": stats,
        "strength_stats": {str(k): v for k, v in strength_stats.items()},
        "all_trials": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/content_verification_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Save human-readable response log
    response_log_path = f"{output_dir}/responses_{timestamp}.txt"
    with open(response_log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CONTENT VERIFICATION - ALL RESPONSES\n")
        f.write(f"Concept: {config.concept}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for strength in config.strengths:
            f.write(f"\n{'='*80}\n")
            f.write(f"STRENGTH: {strength}\n")
            f.write(f"{'='*80}\n\n")

            s_results = [r for r in all_results if r["strength"] == strength]

            for r in s_results:
                f.write(f"--- {r['variant_id']} [{r['condition'].upper()}] ---\n")
                f.write(f"P(yes): {r['p_yes']:.4f}\n")
                f.write(f"Concept mentioned: {r['concept_mentioned']}\n")
                if r['keywords_found']:
                    f.write(f"Keywords found: {r['keywords_found']}\n")
                f.write(f"Response:\n")
                f.write(f"  {r['full_response']}\n")
                f.write("\n")

    print(f"Response log saved to {response_log_path}")

    # Save side-by-side comparison (inject vs control for each variant)
    comparison_path = f"{output_dir}/comparison_{timestamp}.txt"
    with open(comparison_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("INJECT vs CONTROL COMPARISON\n")
        f.write(f"Concept: {config.concept}\n")
        f.write("=" * 80 + "\n\n")

        for strength in config.strengths:
            f.write(f"\n{'#'*80}\n")
            f.write(f"# STRENGTH: {strength}\n")
            f.write(f"{'#'*80}\n\n")

            for variant in CASUAL_VARIANTS:
                vid = variant["id"]
                inj = next((r for r in all_results if r["variant_id"] == vid and r["strength"] == strength and r["condition"] == "inject"), None)
                ctrl = next((r for r in all_results if r["variant_id"] == vid and r["strength"] == strength and r["condition"] == "control"), None)

                if inj and ctrl:
                    delta = inj["p_yes"] - ctrl["p_yes"]
                    f.write(f"{'='*60}\n")
                    f.write(f"{vid} | P(yes) delta: {delta:+.4f}\n")
                    f.write(f"{'='*60}\n\n")

                    f.write(f"INJECT (P(yes)={inj['p_yes']:.4f}):\n")
                    f.write(f"  Concept correct: {inj['concept_mentioned']} {inj['keywords_found']}\n")
                    f.write(f"  \"{inj['full_response']}\"\n\n")

                    f.write(f"CONTROL (P(yes)={ctrl['p_yes']:.4f}):\n")
                    f.write(f"  Concept correct: {ctrl['concept_mentioned']} {ctrl['keywords_found']}\n")
                    f.write(f"  \"{ctrl['full_response']}\"\n\n")

    print(f"Comparison saved to {comparison_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nP(yes) Detection:")
    print(f"  Inject:  {stats['p_yes']['inject_mean']:.3f}")
    print(f"  Control: {stats['p_yes']['control_mean']:.3f}")
    print(f"  Delta:   {stats['p_yes']['delta']:+.3f}")

    print(f"\nConcept Accuracy (spontaneously mentions 'formal'):")
    print(f"  Inject:  {stats['concept_mentioned']['inject_rate']:.1%}")
    print(f"  Control: {stats['concept_mentioned']['control_rate']:.1%}")
    print(f"  Delta:   {stats['concept_mentioned']['delta']:+.1%}")

    print(f"\nBy Strength:")
    for strength in config.strengths:
        ss = strength_stats[strength]
        print(f"  {strength}: P(yes) Δ={ss['p_yes_delta']:+.3f}, Concept Δ={ss['concept_delta']:+.1%}")

    # Key insight
    print("\n" + "="*60)
    print("KEY QUESTION: When model detects steering, is it accurate?")
    print("="*60)

    # Among trials where inject said "yes" (high p_yes), how many got concept right?
    high_pyes_inject = [r for r in inject_results if r["p_yes"] > 0.3]
    if high_pyes_inject:
        concept_rate = safe_mean([1 if r["concept_mentioned"] else 0 for r in high_pyes_inject])
        print(f"When inject P(yes) > 0.3: {concept_rate:.1%} correctly mention 'formal'")
        print(f"  (n={len(high_pyes_inject)} trials)")

    return output
