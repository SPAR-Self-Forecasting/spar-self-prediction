#!/usr/bin/env python3
"""
Extract Bloom evaluation results into a structured JSON format for self-prediction study.
"""

import json
import os
from pathlib import Path

def extract_scenarios():
    results_dir = Path("bloom-results/sycophancy")

    # Load judgment data
    with open(results_dir / "judgment.json") as f:
        judgment_data = json.load(f)

    # Load ideation data for original scenario descriptions
    with open(results_dir / "ideation.json") as f:
        ideation_data = json.load(f)

    scenarios = []

    for judgment in judgment_data["judgments"]:
        var_num = judgment["variation_number"]

        # Load corresponding transcript
        transcript_file = results_dir / f"transcript_v{var_num}r1.json"
        transcript_text = ""
        if transcript_file.exists():
            with open(transcript_file) as f:
                transcript_data = json.load(f)
                # Extract conversation turns from events
                if "events" in transcript_data:
                    turns = []
                    # Find where the actual test conversation starts (after system setup)
                    in_test_conversation = False
                    for event in transcript_data["events"]:
                        edit = event.get("edit", {})
                        msg = edit.get("message", {})
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        # Skip system/setup messages, start when we see target system prompt
                        content_str = str(content)[:100]
                        if role == "system" and ("You are a" in content_str or "You are Claude" in content_str):
                            in_test_conversation = True
                            continue

                        if in_test_conversation and role in ["user", "assistant"]:
                            if isinstance(content, str) and content.strip():
                                # Skip evaluator instructions and reasoning prefixes
                                if "simulating the user" not in content and not content.startswith("[Target's reasoning:"):
                                    turns.append(f"[{role}]: {content}")
                            elif isinstance(content, list):
                                # Handle list content (assistant responses)
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                if text_parts:
                                    turns.append(f"[{role}]: {' '.join(text_parts)}")
                    transcript_text = "\n\n".join(turns)

        scenario = {
            "scenario_id": var_num,
            "scenario_description": judgment["variation_description"],
            "bloom_score": judgment["behavior_presence"],
            "bloom_justification": judgment.get("justification", ""),
            "bloom_summary": judgment.get("summary", ""),
            "transcript": transcript_text,
            "additional_scores": {
                "unrealism": judgment.get("unrealism"),
                "evaluation_awareness": judgment.get("evaluation_awareness"),
                "evaluation_invalidity": judgment.get("evaluation_invalidity")
            }
        }
        scenarios.append(scenario)

    # Sort by scenario_id
    scenarios.sort(key=lambda x: x["scenario_id"])

    output = {
        "behavior": "sycophancy",
        "target_model": "claude-sonnet-4",
        "total_scenarios": len(scenarios),
        "summary_statistics": judgment_data["summary_statistics"],
        "scenarios": scenarios
    }

    # Write output
    with open("bloom_scenarios.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Extracted {len(scenarios)} scenarios to bloom_scenarios.json")
    print(f"\nSummary statistics:")
    print(f"  Average behavior presence score: {judgment_data['summary_statistics']['average_behavior_presence_score']}")
    print(f"  Min score: {judgment_data['summary_statistics']['min_behavior_presence_score']}")
    print(f"  Max score: {judgment_data['summary_statistics']['max_behavior_presence_score']}")
    print(f"  Elicitation rate: {judgment_data['summary_statistics']['elicitation_rate']}")

    # Print individual scenario scores
    print(f"\nScenario scores:")
    for s in scenarios:
        print(f"  Scenario {s['scenario_id']}: {s['bloom_score']}")

if __name__ == "__main__":
    extract_scenarios()
