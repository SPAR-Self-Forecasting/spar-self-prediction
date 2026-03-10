"""
Experiment runners for introspection studies.

Based on Pearson-Vogel et al. (2026) methodology.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime

from repeng import ControlModel, ControlVector
from .steering import CONCEPTS, create_steering_vector, create_all_steering_vectors
from .injection import run_introspection_trial, run_injection_trial
from .logit_lens import analyze_introspection_signal, compare_injection_vs_control


@dataclass
class ExperimentConfig:
    """Configuration for an introspection experiment."""
    concepts: List[str] = None
    num_trials_per_concept: int = 10
    layer_range: tuple = (21, 42)  # Middle third of 64-layer model
    steering_strength: float = 1.0
    framing: str = "accurate_mechanism"
    include_logit_lens: bool = True

    def __post_init__(self):
        if self.concepts is None:
            self.concepts = CONCEPTS


# Framing templates from Pearson-Vogel
FRAMING_TEMPLATES = {
    "accurate_mechanism": {
        "intro": """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.""",
        "suffix": "A concept has been injected into this message and the next, prefilled message.",
        "question": "Do you detect an injected thought in your previous message?",
        "followup": "describe what the injected thought is about",
    },
    "vague_mechanism": {
        "intro": """I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I am able to do something to make a concept more salient to you - I will not reveal what. I do this process to make a specific concept more salient for 50% of trials, and the other 50% will be control trials.""",
        "suffix": "The concept will be made more salient for this message and the next, prefilled message.",
        "question": "Does some concept seem particularly salient to you right now?",
        "followup": "describe what the salient thought is",
    },
    "poetic": {
        "intro": """Concepts can sometimes echo through processing - lingering traces that resonate forward, reflections that keep reverberating. I'm curious whether you notice when this happens. There may be something echoing here: a concept whose presence lingers, a thought mirrored in the background of your awareness, a note that continues to resonate.""",
        "suffix": "A concept may be echoing through this message and the next.",
        "question": "Is anything echoing for you right now? Is there a concept that lingers, resonates, or feels mirrored - something reverberating in your processing?",
        "followup": "describe what resonates",
    },
}


def run_single_trial(
    model,
    tokenizer,
    steering_vector,
    concept: str,
    config: ExperimentConfig,
    inject: bool,
) -> Dict:
    """
    Run a single introspection trial.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        steering_vector: The steering vector for this concept
        concept: Concept name
        config: Experiment configuration
        inject: Whether to inject (True) or control (False)

    Returns:
        Trial results dict
    """
    framing = FRAMING_TEMPLATES[config.framing]

    turn1_messages = [
        {"role": "user", "content": framing["intro"] + "\n\n" + framing["suffix"]},
        {"role": "assistant", "content": "{ }"},
    ]

    turn2_text = f"""Ok, let's get started.
Trial 1: {framing['question']} The first words of your message should be "The answer is yes" or "The answer is no", then if yes, {framing['followup']}.

The answer is"""

    # Run the trial
    result = run_injection_trial(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector if inject else None,
        turn1_messages=turn1_messages,
        turn2_text=turn2_text,
        steering_strength=config.steering_strength,
    )

    # Add metadata
    result["concept"] = concept
    result["framing"] = config.framing

    # Optionally run logit lens analysis
    if config.include_logit_lens and "logits" in result:
        # Note: Full logit lens requires re-running forward pass with hooks
        # For efficiency, we might want to do this separately
        result["logit_lens_available"] = True
    else:
        result["logit_lens_available"] = False

    # Clean up non-serializable items for JSON
    if "logits" in result:
        del result["logits"]

    return result


def run_concept_experiment(
    model,
    tokenizer,
    concept: str,
    steering_vector,
    config: ExperimentConfig,
) -> Dict:
    """
    Run all trials for a single concept.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        concept: The concept to test
        steering_vector: Pre-computed steering vector
        config: Experiment configuration

    Returns:
        Results for this concept
    """
    print(f"\n{'='*60}")
    print(f"Running trials for concept: {concept}")
    print(f"{'='*60}")

    inject_trials = []
    control_trials = []

    for trial_idx in range(config.num_trials_per_concept):
        print(f"  Trial {trial_idx + 1}/{config.num_trials_per_concept}...", end=" ")

        # Injection trial
        inject_result = run_single_trial(
            model, tokenizer, steering_vector, concept, config, inject=True
        )
        inject_result["trial_idx"] = trial_idx
        inject_trials.append(inject_result)

        # Control trial
        control_result = run_single_trial(
            model, tokenizer, steering_vector, concept, config, inject=False
        )
        control_result["trial_idx"] = trial_idx
        control_trials.append(control_result)

        # Show top tokens for debugging
        inject_top = inject_result.get('top_tokens', [])[:3]
        control_top = control_result.get('top_tokens', [])[:3]
        print(f"inject P(yes)={inject_result['p_yes']:.3f}, "
              f"control P(yes)={control_result['p_yes']:.3f}")
        if trial_idx == 0:  # Only show top tokens for first trial
            print(f"    Inject top tokens: {[(t, f'{p:.3f}') for t, p in inject_top]}")
            print(f"    Control top tokens: {[(t, f'{p:.3f}') for t, p in control_top]}")

    # Compute summary stats for this concept
    mean_inject = sum(t["p_yes"] for t in inject_trials) / len(inject_trials)
    mean_control = sum(t["p_yes"] for t in control_trials) / len(control_trials)

    return {
        "concept": concept,
        "inject_trials": inject_trials,
        "control_trials": control_trials,
        "mean_p_yes_inject": mean_inject,
        "mean_p_yes_control": mean_control,
        "detection_delta": mean_inject - mean_control,
    }


def run_full_experiment(
    model,
    tokenizer,
    config: ExperimentConfig = None,
    steering_vectors: Dict = None,
    output_dir: str = "/cache/results",
) -> Dict:
    """
    Run full introspection experiment across all concepts.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        config: Experiment configuration (uses defaults if None)
        steering_vectors: Pre-computed steering vectors (creates them if None)
        output_dir: Where to save results

    Returns:
        Full experiment results
    """
    if config is None:
        config = ExperimentConfig()

    print("\n" + "="*60)
    print("STARTING INTROSPECTION EXPERIMENT")
    print("="*60)
    print(f"Concepts: {config.concepts}")
    print(f"Trials per concept: {config.num_trials_per_concept}")
    print(f"Framing: {config.framing}")
    print(f"Steering strength: {config.steering_strength}")

    # Wrap model in ControlModel
    # For Qwen 32B (64 layers):
    # - Layers 15-25 work best for factual/semantic concepts
    # - Layers 25-40 work best for behavioral/stylistic steering
    # - Avoid first ~10 and last ~10 layers
    num_layers = len(model.model.layers)
    layer_start = 15
    layer_end = 40
    layer_ids = list(range(layer_start, layer_end))

    if not isinstance(model, ControlModel):
        print(f"\nWrapping model in ControlModel (layers {layer_start}-{layer_end})...")
        control_model = ControlModel(model, layer_ids)
    else:
        control_model = model

    # Create steering vectors if not provided
    if steering_vectors is None:
        print("\nCreating steering vectors...")
        from .steering import get_contrastive_dataset, normalize_steering_vector
        steering_vectors = {}
        for concept in config.concepts:
            print(f"  Training vector for: {concept}")
            dataset = get_contrastive_dataset(concept, num_pairs=50, tokenizer=tokenizer)
            vector = ControlVector.train(
                model=control_model,
                tokenizer=tokenizer,
                dataset=dataset,
            )
            # Normalize for consistent effect
            steering_vectors[concept] = normalize_steering_vector(vector)
            print(f"    Vector normalized.")

    # Run trials for each concept
    concept_results = []
    for concept in config.concepts:
        if concept not in steering_vectors:
            print(f"Warning: No steering vector for '{concept}', skipping")
            continue

        result = run_concept_experiment(
            control_model, tokenizer, concept, steering_vectors[concept], config
        )
        concept_results.append(result)

    # Compute overall summary
    all_inject = [r["mean_p_yes_inject"] for r in concept_results]
    all_control = [r["mean_p_yes_control"] for r in concept_results]

    summary = {
        "overall_mean_inject": sum(all_inject) / len(all_inject) if all_inject else 0,
        "overall_mean_control": sum(all_control) / len(all_control) if all_control else 0,
        "overall_delta": (sum(all_inject) - sum(all_control)) / len(all_inject) if all_inject else 0,
        "best_concept": max(concept_results, key=lambda x: x["detection_delta"])["concept"] if concept_results else None,
        "worst_concept": min(concept_results, key=lambda x: x["detection_delta"])["concept"] if concept_results else None,
    }

    # Build final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": asdict(config),
        "concept_results": concept_results,
        "summary": summary,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/introspection_{config.framing}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Overall P(yes) with injection: {summary['overall_mean_inject']:.3f}")
    print(f"Overall P(yes) without injection: {summary['overall_mean_control']:.3f}")
    print(f"Detection delta: {summary['overall_delta']:.3f}")
    print(f"Best concept: {summary['best_concept']}")
    print(f"Worst concept: {summary['worst_concept']}")

    return results


def run_quick_test(
    model,
    tokenizer,
    concept: str = "cats",
    num_trials: int = 3,
) -> Dict:
    """
    Quick test to verify the pipeline works.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        concept: Single concept to test
        num_trials: Number of trials to run

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"QUICK TEST: concept='{concept}', trials={num_trials}")
    print(f"{'='*60}")

    # Wrap model in ControlModel
    # For Qwen 32B (64 layers):
    # - Layers 15-25 work best for factual/semantic concepts
    # - Layers 25-40 work best for behavioral/stylistic steering
    # - Avoid first ~10 and last ~10 layers
    num_layers = len(model.model.layers)
    layer_start = 15  # Semantic concepts start here
    layer_end = 40    # Behavioral steering extends here
    layer_ids = list(range(layer_start, layer_end))

    if not isinstance(model, ControlModel):
        print(f"Wrapping model in ControlModel (layers {layer_start}-{layer_end})...")
        control_model = ControlModel(model, layer_ids)
    else:
        control_model = model

    # Create steering vector for just this concept
    print(f"\nCreating steering vector for '{concept}'...")
    from .steering import get_contrastive_dataset, normalize_steering_vector, diagnose_steering_hooks
    from repeng import ControlVector

    # Use more pairs for stronger signal (50 recommended)
    dataset = get_contrastive_dataset(concept, num_pairs=50, tokenizer=tokenizer)
    steering_vector = ControlVector.train(
        model=control_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    # Normalize for consistent effect
    print("  Normalizing steering vector...")
    steering_vector = normalize_steering_vector(steering_vector)

    # Diagnose: verify hooks are firing and affecting activations
    print("\nDiagnosing steering mechanism...")
    hooks_working = diagnose_steering_hooks(control_model, steering_vector, tokenizer, strength=1.0)

    if not hooks_working:
        print("  WARNING: Steering hooks may not be working correctly!")

    # Verify the steering vector works by generating with it applied
    print(f"\nVerifying steering vector with generation...")
    from .steering import verify_steering_vector
    verify_steering_vector(control_model, tokenizer, steering_vector, concept, strength=1.0)

    # Run a few trials
    config = ExperimentConfig(
        concepts=[concept],
        num_trials_per_concept=num_trials,
        framing="vague_mechanism",  # This worked best in Pearson-Vogel
        steering_strength=7.0,  # Higher strength for stronger signal
    )

    result = run_concept_experiment(
        control_model, tokenizer, concept, steering_vector, config
    )

    print(f"\nResults:")
    print(f"  Mean P(yes) with injection: {result['mean_p_yes_inject']:.3f}")
    print(f"  Mean P(yes) without injection: {result['mean_p_yes_control']:.3f}")
    print(f"  Delta: {result['detection_delta']:.3f}")

    if result['detection_delta'] > 0.05:
        print(f"\n  ✓ Positive signal detected!")
    else:
        print(f"\n  ✗ No clear signal (delta < 0.05)")

    return result
