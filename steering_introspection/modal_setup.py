"""
Modal infrastructure for steering vector introspection experiments.

Usage:
    modal run modal_setup.py::test_setup           # Verify GPU + model works
    modal run modal_setup.py::download_model       # Pre-cache model weights
    modal run modal_setup.py::quick_test           # Quick sanity check (few trials)
    modal run modal_setup.py::run_experiment       # Full experiment (single framing)
    modal run modal_setup.py::run_robust_experiment  # Robust experiment with prompt variants
"""

import modal

# Create persistent volume for model weights and results
volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "numpy",
        "pandas",
        "matplotlib",
    )
    .pip_install("git+https://github.com/vgel/repeng.git")
    # Copy our source code into the image
    .add_local_dir("src", "/root/src")
)

app = modal.App("steering-introspection", image=image)

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_CACHE_PATH = "/cache/models"


@app.function(
    volumes={"/cache": volume},
    timeout=1800,
)
def download_model():
    """Pre-download model weights to volume (run once)."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {MODEL_CACHE_PATH}...")
    snapshot_download(
        MODEL_ID,
        local_dir=f"{MODEL_CACHE_PATH}/{MODEL_ID.replace('/', '_')}",
    )

    volume.commit()
    print("Done! Model cached for future runs.")


def load_model():
    """Helper to load model from cache."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = f"{MODEL_CACHE_PATH}/{MODEL_ID.replace('/', '_')}"

    print(f"Loading model from {local_path}...")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded! Device: {model.device}")

    return model, tokenizer


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=3600,
)
def test_setup():
    """Verify GPU access and model loading works."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = load_model()
    print(f"Model layers: {len(model.model.layers)}")

    # Quick generation test
    inputs = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"Test generation: {tokenizer.decode(outputs[0])}")

    return "Setup verified!"


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=3600,
)
def quick_test(concept: str = "cats", num_trials: int = 3):
    """
    Quick test to verify the steering + introspection pipeline works.

    Args:
        concept: Which concept to test
        num_trials: Number of trials to run
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.experiments import run_quick_test

    result = run_quick_test(model, tokenizer, concept, num_trials)

    volume.commit()
    return result


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=14400,  # 4 hours
)
def run_experiment(
    concepts: str = None,  # Comma-separated, e.g., "cats,fear,love"
    num_trials: int = 10,
    framing: str = "vague_mechanism",
    steering_strength: float = 1.0,
):
    """
    Run full introspection experiment.

    Args:
        concepts: Comma-separated concept list (None = all 9 concepts)
        num_trials: Trials per concept
        framing: "accurate_mechanism", "vague_mechanism", or "poetic"
        steering_strength: How strongly to inject (default 1.0)
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.experiments import run_full_experiment, ExperimentConfig

    # Parse concepts
    if concepts:
        concept_list = [c.strip() for c in concepts.split(",")]
    else:
        concept_list = None  # Use all defaults

    config = ExperimentConfig(
        concepts=concept_list,
        num_trials_per_concept=num_trials,
        framing=framing,
        steering_strength=steering_strength,
    )

    results = run_full_experiment(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir="/cache/results",
    )

    volume.commit()

    return {
        "summary": results["summary"],
        "output_path": f"/cache/results/introspection_{framing}_{results['timestamp']}.json",
    }


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=3600,
)
def create_and_save_vectors(concepts: str = None):
    """
    Create steering vectors and save to volume for reuse.

    Args:
        concepts: Comma-separated concept list (None = all 9 concepts)
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.steering import create_all_steering_vectors, save_steering_vectors, CONCEPTS

    # Parse concepts
    if concepts:
        concept_list = [c.strip() for c in concepts.split(",")]
    else:
        concept_list = CONCEPTS

    vectors = create_all_steering_vectors(model, tokenizer, concept_list)

    save_path = "/cache/steering_vectors.pkl"
    save_steering_vectors(vectors, save_path)

    volume.commit()

    return f"Saved {len(vectors)} steering vectors to {save_path}"


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=14400,  # 4 hours
)
def run_robust_experiment(
    concept: str = "formal",
    strengths: str = "1.0,5.0,7.0",  # Comma-separated
    variants: str = None,  # Comma-separated variant IDs, None = all 10
):
    """
    Run robust experiment with prompt variants for real statistical variance.

    Each prompt variant is an independent sample, giving defensible error bars.

    Args:
        concept: Which concept to test (default: formal)
        strengths: Comma-separated steering strengths to test
        variants: Comma-separated variant IDs (None = all 10 variants)

    Usage:
        modal run modal_setup.py::run_robust_experiment
        modal run modal_setup.py::run_robust_experiment --concept formal --strengths "1.0,5.0,10.0"
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.robust_experiment import run_robust_experiment as run_robust, RobustExperimentConfig

    # Parse inputs
    strength_list = [float(s.strip()) for s in strengths.split(",")]
    variant_list = [v.strip() for v in variants.split(",")] if variants else None

    config = RobustExperimentConfig(
        concept=concept,
        strengths=strength_list,
        variants=variant_list,
    )

    results = run_robust(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir="/cache/results",
    )

    volume.commit()

    return {
        "statistics": results["statistics"],
        "strength_breakdown": results["strength_breakdown"],
        "output_path": f"/cache/results/robust_{concept}_{results['timestamp']}.json",
    }


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=14400,  # 4 hours
)
def run_casual_experiment(
    concept: str = "formal",
    strengths: str = "1.0,5.0,7.0",
):
    """
    Run experiment on 20 casual prompt variants.

    20 variants × 3 strengths = 60 conditions with proper error bars.

    Args:
        concept: Which concept to steer (default: formal)
        strengths: Comma-separated steering strengths

    Usage:
        modal run modal_setup.py::run_casual_experiment
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.casual_experiment import run_casual_experiment as run_casual, CasualExperimentConfig

    strength_list = [float(s.strip()) for s in strengths.split(",")]

    config = CasualExperimentConfig(
        concept=concept,
        strengths=strength_list,
    )

    results = run_casual(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir="/cache/results",
    )

    volume.commit()

    return {
        "overall_stats": results["overall_stats"],
        "strength_stats": results["strength_stats"],
        "output_path": f"/cache/results/casual_variants_{results['timestamp']}.json",
    }


@app.function(
    gpu="A100-80GB",
    volumes={"/cache": volume},
    timeout=14400,  # 4 hours
)
def run_content_experiment(
    concept: str = "formal",
    strengths: str = "1.0,5.0,7.0",
):
    """
    Run content verification experiment.

    Tests whether models correctly identify the steered concept.
    Uses original 20 casual prompts - no hints given to model.

    Key question: When steered toward "formal", does model spontaneously
    say "formal/professional" or something random?

    Args:
        concept: Which concept to steer (default: formal)
        strengths: Comma-separated steering strengths

    Usage:
        modal run modal_setup.py::run_content_experiment
    """
    import sys
    sys.path.insert(0, "/root")

    model, tokenizer = load_model()

    from src.content_experiment import run_content_experiment as run_content, ContentExperimentConfig

    strength_list = [float(s.strip()) for s in strengths.split(",")]

    config = ContentExperimentConfig(
        concept=concept,
        strengths=strength_list,
    )

    results = run_content(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir="/cache/results",
    )

    volume.commit()

    return {
        "stats": results["stats"],
        "strength_stats": results["strength_stats"],
        "output_path": f"/cache/results/content_verification_{results['timestamp']}.json",
    }


@app.local_entrypoint()
def main():
    """Run from command line: modal run modal_setup.py"""
    print("Running quick test...")
    result = quick_test.remote()
    print(f"Result: {result}")
