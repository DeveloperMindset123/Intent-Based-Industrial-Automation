#!/usr/bin/env python3
"""Convert my_scenarios.json to a clean JSONL format for Hugging Face."""

import json
from pathlib import Path

def standardize_procedure(procedure):
    """Convert procedure to consistent object format."""
    if isinstance(procedure, str):
        return {
            "problem": procedure,
            "steps": ""
        }
    return procedure

def main():
    input_file = Path(__file__).parent / "my_scenarios.json"
    output_file = Path(__file__).parent / "my_scenarios_clean.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("pdm_scenarios", [])
    generation_info = data.get("generation_info", {})

    print(f"Found {len(scenarios)} scenarios")
    print(f"Generation info: {generation_info}")

    # Count procedure types before conversion
    string_procedures = sum(1 for s in scenarios if isinstance(s.get("procedure"), str))
    object_procedures = sum(1 for s in scenarios if isinstance(s.get("procedure"), dict))
    print(f"Procedure types - String: {string_procedures}, Object: {object_procedures}")

    # Write JSONL with standardized procedures
    with open(output_file, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            # Standardize procedure field
            if "procedure" in scenario:
                scenario["procedure"] = standardize_procedure(scenario["procedure"])

            # Write one JSON object per line
            f.write(json.dumps(scenario, ensure_ascii=False) + "\n")

    print(f"Written {len(scenarios)} scenarios to {output_file}")
    print("Done! File is ready for Hugging Face upload.")

if __name__ == "__main__":
    main()
