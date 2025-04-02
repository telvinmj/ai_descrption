"""
Test the uncertain models API endpoint directly.
"""

import os
import sys
import json
import requests

def main():
    # Set the API endpoint URL
    url = "http://localhost:8000/api/models/uncertain/"
    
    print(f"Testing uncertain models API endpoint: {url}")
    
    try:
        # Make a GET request to the uncertain models endpoint
        response = requests.get(url)
        
        # Print the response status code
        print(f"Response status code: {response.status_code}")
        
        # Print the response headers
        print("\nResponse headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            print("\nResponse data:")
            print(json.dumps(data, indent=2))
            
            # Check if there are any models
            if "models" in data and "count" in data:
                count = data["count"]
                models = data["models"]
                print(f"\nFound {count} uncertain models.")
                
                if count > 0:
                    print("\nModel names:")
                    for model in models:
                        print(f"  - {model['name']} (confidence: {model.get('ai_confidence_score', 'N/A')})")
                else:
                    print("\nNo uncertain models found.")
            else:
                print("\nUnexpected response format. No 'models' or 'count' field found.")
        else:
            print(f"\nError: Failed to get data from API. Status code: {response.status_code}")
            print("Response body:")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {url}. Is the backend server running?")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 