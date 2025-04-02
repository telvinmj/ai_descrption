"""
Test script to create models with low confidence scores to test the uncertain models functionality.
"""

import json
import os
import sys
import random
from pathlib import Path

def main():
    # Path to the metadata file
    metadata_path = Path("exports/uni_metadata.json")
    
    if not metadata_path.exists():
        print(f"Metadata file not found at {metadata_path}")
        return 1
    
    # Load the metadata
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Count original models
    original_count = len(metadata.get("models", []))
    print(f"Original metadata contains {original_count} models")
    
    # Count models with low confidence (2 or below)
    low_confidence_count = 0
    for model in metadata.get("models", []):
        if model.get("ai_confidence_score", 0) <= 2:
            low_confidence_count += 1
    
    print(f"Currently there are {low_confidence_count} models with low confidence (2 or below)")
    
    # Update at least 5 models to have low confidence scores
    modified_count = 0
    for i, model in enumerate(metadata.get("models", [])):
        if i < 5:
            model["ai_confidence_score"] = random.choice([1, 2])
            print(f"Set model {model['name']} confidence score to {model['ai_confidence_score']}")
            modified_count += 1
    
    # Save the updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated {modified_count} models to have low confidence scores (2 or below)")
    print(f"Saved updated metadata to {metadata_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 