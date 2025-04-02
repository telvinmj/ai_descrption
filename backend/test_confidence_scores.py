"""
Test script for AI Description Service confidence score summary.
This script will create a small test dataset and process it using the AI Description Service
to verify that confidence scores are properly displayed in the summary.
"""

import json
import os
import sys
import random
from services.ai_description_service import AIDescriptionService

def main():
    # Initialize the service
    print("Initializing AIDescriptionService...")
    service = AIDescriptionService()
    
    # Create a test metadata file with sample models
    print("Creating test metadata...")
    test_metadata = {
        "models": [
            {
                "name": "test_model_1",
                "project": "test_project",
                "description": "",
                "columns": [
                    {"name": "id", "type": "integer", "description": ""},
                    {"name": "name", "type": "string", "description": ""},
                    {"name": "created_at", "type": "timestamp", "description": ""}
                ]
            },
            {
                "name": "test_model_2",
                "project": "test_project",
                "description": "",
                "columns": [
                    {"name": "user_id", "type": "integer", "description": ""},
                    {"name": "product_id", "type": "integer", "description": ""},
                    {"name": "quantity", "type": "integer", "description": ""},
                    {"name": "price", "type": "float", "description": ""}
                ]
            }
        ]
    }
    
    # Create a small batch to test the batch processing function
    test_batch = test_metadata["models"]
    
    try:
        # Since we might not have an API key, let's mock confidence scores
        print("Testing with mocked confidence scores since API key may not be available...")
        
        # Add mock descriptions and confidence scores
        for model in test_batch:
            model_name = model.get("name", "Unknown")
            
            # Add mock model description and confidence
            model["description"] = f"This is a test description for {model_name}"
            model["ai_description"] = f"This is a test AI description for {model_name}"
            model["ai_confidence_score"] = random.randint(1, 5)
            model["user_edited"] = False
            
            # Add mock column descriptions and confidence scores
            for column in model.get("columns", []):
                col_name = column.get("name", "Unknown")
                column["description"] = f"This is a description for column {col_name}"
                column["ai_description"] = f"This is an AI description for column {col_name}"
                column["ai_confidence_score"] = random.randint(1, 5)
                column["user_edited"] = False
        
        # Test the batch processing summary directly
        print("\n=== Testing batch processing summary with mock data ===")
        # This will simulate batch completion and confidence score reporting
        service._test_batch_confidence_reporting(test_batch)
        
        # Verify confidence scores
        print("\nVerifying confidence scores in mock data...")
        for model in test_batch:
            confidence = model.get("ai_confidence_score")
            print(f"Model {model['name']} confidence score: {confidence}")
            
            # Check columns
            for column in model.get("columns", []):
                col_confidence = column.get("ai_confidence_score")
                print(f"  Column {column['name']} confidence score: {col_confidence}")
        
        # Save the processed metadata
        output_path = "test_output.json"
        with open(output_path, "w") as f:
            json.dump(test_batch, f, indent=2)
        print(f"\nTest data saved to {output_path}")
        
        print("\nTest completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())