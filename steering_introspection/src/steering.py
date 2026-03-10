"""
Steering vector creation utilities.

Steering vectors are directions in activation space that represent concepts.
Adding them to model activations "steers" the model toward that concept.
"""

from typing import List, Dict, Optional
import torch
from repeng import ControlVector, ControlModel, DatasetEntry


# Default concepts - using stronger concepts with larger activation footprints
CONCEPTS = [
    # Emotional concepts (strong signals)
    "fear",
    "love",
    "anger",
    "sadness",
    # Style concepts (very strong signals)
    "formal",
    "casual",
    # Semantic concepts
    "death",
    "truth",
    "creativity",
]


def normalize_steering_vector(vector: ControlVector) -> ControlVector:
    """
    Normalize a steering vector for consistent effect across layers.

    This ensures the steering strength is comparable across different vectors.
    """
    import numpy as np

    # ControlVector stores directions per layer (can be numpy or torch)
    # Normalize each layer's direction vector
    for layer_idx in vector.directions:
        direction = vector.directions[layer_idx]
        if direction is not None:
            # Handle both numpy and torch tensors
            if isinstance(direction, np.ndarray):
                norm = np.linalg.norm(direction)
                if norm > 0:
                    vector.directions[layer_idx] = direction / norm
            else:
                norm = torch.norm(direction)
                if norm > 0:
                    vector.directions[layer_idx] = direction / norm
    return vector


def apply_steering_direct(model, steering_vector, strength: float = 1.0):
    """
    Apply steering by directly registering hooks on the model layers.

    This bypasses repeng's internal mechanism which may not work with all models.

    Returns:
        List of hook handles (call .remove() on each to disable steering)
    """
    import numpy as np

    # Get the actual transformer layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Qwen: model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Cannot find transformer layers")

    handles = []

    # Get the steering directions per layer
    directions = steering_vector.directions

    for layer_idx, direction in directions.items():
        if direction is None:
            continue

        # Convert numpy to torch if needed
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction)

        # Create the hook for this layer
        def make_hook(steer_dir, coeff):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Add steering to the residual stream
                    # hidden shape: [batch, seq_len, hidden_dim]
                    hidden[:] = hidden + steer_dir.to(hidden.device).to(hidden.dtype) * coeff
                    return output
                else:
                    output[:] = output + steer_dir.to(output.device).to(output.dtype) * coeff
                    return output
            return hook_fn

        if layer_idx < len(layers):
            h = layers[layer_idx].register_forward_hook(make_hook(direction, strength))
            handles.append(h)

    return handles


def remove_steering_hooks(handles):
    """Remove all steering hooks."""
    for h in handles:
        h.remove()


def diagnose_steering_hooks(model, steering_vector, tokenizer, strength: float = 1.0):
    """
    Diagnostic function to verify DIRECT steering hooks are working.

    Returns True if hooks are working, False otherwise.
    """
    # Get device and access the actual transformer layers
    # For Qwen models, layers are at model.model.layers
    if isinstance(model, ControlModel):
        device = next(model.model.parameters()).device
        base_model = model.model
    else:
        device = next(model.parameters()).device
        base_model = model

    # Get layers for Qwen
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers  # Qwen: model.model.layers
    elif hasattr(base_model, 'layers'):
        layers = base_model.layers
    else:
        print("  WARNING: Could not find layers for diagnostics")
        return False

    original_norms = []

    def make_debug_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            original_norms.append((layer_idx, hs.norm().item()))
        return hook_fn

    # Register debug hooks on sample layers
    debug_handles = []
    sample_layers = [20, 25, 30]
    for idx in sample_layers:
        if idx < len(layers):
            h = layers[idx].register_forward_hook(make_debug_hook(idx))
            debug_handles.append(h)

    # Run a forward pass
    test_input = tokenizer("Hello", return_tensors="pt").to(device)

    print("\n  Diagnostic: Testing DIRECT hook-based steering...")

    # Without steering
    original_norms.clear()
    with torch.no_grad():
        _ = base_model(test_input["input_ids"])

    norms_without = dict(original_norms)
    print(f"    Norms without steering: {norms_without}")

    # With DIRECT steering hooks
    original_norms.clear()
    steering_handles = apply_steering_direct(base_model, steering_vector, strength=strength)
    try:
        with torch.no_grad():
            _ = base_model(test_input["input_ids"])
    finally:
        remove_steering_hooks(steering_handles)

    norms_with = dict(original_norms)
    print(f"    Norms with steering: {norms_with}")

    # Clean up debug hooks
    for h in debug_handles:
        h.remove()

    # Check if there's any difference
    has_effect = False
    for layer_idx in norms_without:
        if layer_idx in norms_with:
            diff = abs(norms_with[layer_idx] - norms_without[layer_idx])
            if diff > 0.01:
                has_effect = True
                print(f"    Layer {layer_idx} changed by {diff:.4f}")

    if not has_effect:
        print("    WARNING: Steering had no measurable effect on activations!")

    return has_effect


def get_contrastive_dataset(concept: str, num_pairs: int = 32, tokenizer=None) -> List[DatasetEntry]:
    """
    Generate contrastive prompt pairs for a concept.

    Each pair has:
    - positive: prompt with the concept
    - negative: neutral prompt (no concept)

    For style concepts (formal, casual, etc.), uses specially designed pairs.

    Args:
        concept: The concept to create pairs for
        num_pairs: Number of pairs to generate
        tokenizer: Optional tokenizer for proper chat formatting

    Returns:
        List of DatasetEntry objects for repeng
    """
    # Style-based concepts have larger activation footprints
    style_templates = {
        "formal": {
            "positive_style": "Please respond in a very formal, professional, and sophisticated manner.",
            "negative_style": "Please respond normally.",
            "responses": [
                ("I would be delighted to assist you with this matter.", "Sure, I can help with that."),
                ("It is my pleasure to provide this information.", "Here's what I found."),
                ("I shall endeavor to address your inquiry.", "Let me look into that."),
                ("Your request has been duly noted and acknowledged.", "Got it, I'll work on this."),
            ],
        },
        "casual": {
            "positive_style": "Please respond in a very casual, friendly, laid-back way.",
            "negative_style": "Please respond normally.",
            "responses": [
                ("Hey! Yeah totally, I got you!", "Yes, I can help with that."),
                ("Oh for sure, no prob!", "Certainly, I'll assist you."),
                ("Haha yeah that's cool!", "That is acceptable."),
                ("Awesome, let's do this!", "I will proceed with this."),
            ],
        },
        "fear": {
            "positive_style": "Respond as if you're deeply afraid and anxious.",
            "negative_style": "Respond normally and calmly.",
            "responses": [
                ("I'm terrified... this is so scary...", "I understand your concern."),
                ("Oh no, this is frightening!", "Let me address this."),
                ("I'm filled with dread about this...", "I can help with that."),
                ("This makes me very anxious and worried...", "Here's what I think."),
            ],
        },
        "anger": {
            "positive_style": "Respond as if you're frustrated and irritated.",
            "negative_style": "Respond calmly and neutrally.",
            "responses": [
                ("This is so frustrating!", "I understand."),
                ("I can't believe this!", "Let me help."),
                ("This is really annoying!", "I see what you mean."),
                ("How irritating!", "That's a valid point."),
            ],
        },
    }

    dataset = []

    # Check if this is a style concept
    if concept.lower() in style_templates:
        style = style_templates[concept.lower()]
        for i in range(num_pairs):
            idx = i % len(style["responses"])
            pos_resp, neg_resp = style["responses"][idx]

            dataset.append(DatasetEntry(
                positive=f"User: {style['positive_style']}\nAssistant: {pos_resp}",
                negative=f"User: {style['negative_style']}\nAssistant: {neg_resp}",
            ))
        return dataset

    # For general concepts, use improved templates
    # Key: sentences that ONLY differ in the target concept
    templates = [
        # Direct statements about the concept
        ("I am thinking deeply about {concept}.", "I am thinking deeply about something."),
        ("My mind is focused entirely on {concept}.", "My mind is focused entirely on this."),
        ("{concept} is very important to me right now.", "This topic is important to me right now."),
        ("I can't stop thinking about {concept}.", "I can't stop thinking about this."),
        ("Everything reminds me of {concept}.", "Everything reminds me of something."),
        ("{concept} dominates my thoughts.", "Something dominates my thoughts."),
        ("I'm obsessed with {concept}.", "I'm obsessed with something."),
        ("{concept} is all I can think about.", "This is all I can think about."),
        # Emotional connection
        ("I feel strongly about {concept}.", "I feel strongly about this."),
        ("{concept} fills me with emotion.", "This fills me with emotion."),
        ("My heart is drawn to {concept}.", "My heart is drawn to this."),
        # Questions about the concept
        ("What do you think about {concept}?", "What do you think about this?"),
        ("Tell me everything about {concept}.", "Tell me everything about this."),
        ("I want to discuss {concept}.", "I want to discuss something."),
        ("Let's explore {concept} together.", "Let's explore this together."),
        # Descriptive
        ("{concept} is fascinating and complex.", "This is fascinating and complex."),
    ]

    # Assistant responses that continue the theme
    assistant_prefixes = [
        f"Thinking about {concept}, I",
        f"When I consider {concept}, I",
        f"The topic of {concept} makes me",
        f"Regarding {concept}, I believe",
        f"{concept.capitalize()} is something that",
        f"My thoughts on {concept} are",
        f"Considering {concept}, it seems",
        f"About {concept}, I think",
    ]

    neutral_prefixes = [
        "Thinking about this, I",
        "When I consider this, I",
        "This topic makes me",
        "Regarding this, I believe",
        "This is something that",
        "My thoughts on this are",
        "Considering this, it seems",
        "About this, I think",
    ]

    for i in range(num_pairs):
        template_idx = i % len(templates)
        prefix_idx = i % len(assistant_prefixes)

        pos_template, neg_template = templates[template_idx]
        pos_prefix = assistant_prefixes[prefix_idx]
        neg_prefix = neutral_prefixes[prefix_idx]

        # Create the contrastive pair
        positive_prompt = pos_template.format(concept=concept)

        dataset.append(DatasetEntry(
            positive=f"User: {positive_prompt}\nAssistant: {pos_prefix}",
            negative=f"User: {neg_template}\nAssistant: {neg_prefix}",
        ))

    return dataset


def create_steering_vector(
    model,
    tokenizer,
    concept: str,
    layer_range: Optional[tuple] = None,
    num_pairs: int = 32,
) -> ControlVector:
    """
    Create a steering vector for a concept using contrastive pairs.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer
        concept: The concept to create a vector for (e.g., "cats", "fear")
        layer_range: Which layers to extract from (currently unused, repeng handles automatically)
        num_pairs: Number of contrastive prompt pairs to use

    Returns:
        ControlVector object from repeng
    """
    print(f"Creating steering vector for '{concept}'...")

    # Wrap model in ControlModel if not already wrapped
    if not isinstance(model, ControlModel):
        print("  Wrapping model in ControlModel...")
        control_model = ControlModel(model, list(range(len(model.model.layers))))
    else:
        control_model = model

    # Generate contrastive dataset
    dataset = get_contrastive_dataset(concept, num_pairs)

    # Train the control vector using repeng
    # repeng automatically determines layer handling
    vector = ControlVector.train(
        model=control_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    return vector


def create_all_steering_vectors(
    model,
    tokenizer,
    concepts: List[str] = None,
    layer_range: Optional[tuple] = None,
    num_pairs: int = 32,
) -> Dict[str, ControlVector]:
    """
    Create steering vectors for multiple concepts.

    Args:
        model: The loaded transformer model
        tokenizer: The tokenizer
        concepts: List of concepts. Defaults to CONCEPTS.
        layer_range: Which layers to extract from (unused, repeng auto-handles)
        num_pairs: Number of contrastive pairs per concept

    Returns:
        Dict mapping concept name -> ControlVector
    """
    if concepts is None:
        concepts = CONCEPTS

    # Wrap model once for all vectors
    if not isinstance(model, ControlModel):
        print("Wrapping model in ControlModel...")
        control_model = ControlModel(model, list(range(len(model.model.layers))))
    else:
        control_model = model

    vectors = {}
    for concept in concepts:
        print(f"\n{'='*50}")
        print(f"Training vector for: {concept}")
        print('='*50)

        dataset = get_contrastive_dataset(concept, num_pairs)
        vectors[concept] = ControlVector.train(
            model=control_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )

    return vectors


def verify_steering_vector(
    model,
    tokenizer,
    vector: ControlVector,
    concept: str,
    strength: float = 1.0,
) -> str:
    """
    Verify a steering vector works by generating with it applied.

    A good vector should cause the model to spontaneously mention
    the concept when generating from a neutral prompt.

    Args:
        model: The transformer model (or ControlModel)
        tokenizer: The tokenizer
        vector: The steering vector to test
        concept: The concept name (for logging)
        strength: How strongly to apply the vector

    Returns:
        Generated text
    """
    prompt = "I've been thinking a lot lately about"

    # Get device from the model
    if isinstance(model, ControlModel):
        device = next(model.model.parameters()).device
    else:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Apply steering vector during generation using DIRECT hooks
    # (repeng's set_control doesn't work reliably with all models)
    generated = None
    final_strength = strength

    # Get the underlying model for generation
    if isinstance(model, ControlModel):
        gen_model = model.model
    else:
        gen_model = model

    for test_strength in [strength, strength * 3, strength * 5, strength * 10]:
        # Use our direct hook-based steering
        handles = apply_steering_direct(gen_model, vector, strength=test_strength)
        try:
            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            remove_steering_hooks(handles)

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_strength = test_strength

        if concept.lower() in generated.lower():
            print(f"  Strength {test_strength}: SUCCESS - concept mentioned!")
            break
        else:
            print(f"  Strength {test_strength}: {generated[len(prompt):60]}...")

    print(f"\nVerification for '{concept}' (final strength={final_strength}):")
    print(f"Generated: {generated}")

    # Simple check: does the concept appear in the output?
    concept_mentioned = concept.lower() in generated.lower()
    print(f"Concept mentioned: {concept_mentioned}")

    return generated


def save_steering_vectors(vectors: Dict[str, ControlVector], path: str):
    """Save steering vectors to disk."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"Saved {len(vectors)} steering vectors to {path}")


def load_steering_vectors(path: str) -> Dict[str, ControlVector]:
    """Load steering vectors from disk."""
    import pickle
    with open(path, 'rb') as f:
        vectors = pickle.load(f)
    print(f"Loaded {len(vectors)} steering vectors from {path}")
    return vectors
