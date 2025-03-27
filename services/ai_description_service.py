import os
import json
import requests
from typing import Dict, Any, List, Optional

class AIDescriptionService:
    """Service for generating AI descriptions for models and columns using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
            print("Set it using: export GEMINI_API_KEY='your-api-key'")
        else:
            print(f"Using Gemini API key: {self.api_key[:5]}...{self.api_key[-4:]}")
        
        # Updated to use gemini-2.0-flash-lite model as specified
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite:generateContent"
    
    def _make_api_request(self, prompt: str) -> Optional[str]:
        """Make a request to Gemini API to generate content"""
        if not self.api_key:
            print("Error: Gemini API key not set")
            return None
        
        try:
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 200,
                }
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error from Gemini API: {response.status_code} - {response.text}")
                return None
            
            # Parse the response
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        # Return the full text without any truncation
                        description = parts[0]["text"].strip()
                        # Check for truncation indicators and log a warning if found
                        if description.endswith('...') or description.endswith('…'):
                            print(f"Warning: AI description appears to be truncated: {description}")
                        return description
            
            print("Unexpected response format from Gemini API")
            return None
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None
    
    def generate_column_description(self, column_name: str, model_name: str, sql_context: str = None, 
                                   column_type: str = None, table_context: str = None) -> Optional[str]:
        """Generate a description for a column based on its name and context"""
        # Build a detailed prompt with all available context
        prompt = f"""
        As a database expert, provide a concise, accurate, and helpful description (2-3 sentences) for a column in a database table.

        Column Name: {column_name}
        Model/Table Name: {model_name}
        Column Data Type: {column_type or 'Unknown'}
        """
        
        # Add table context if available
        if table_context:
            prompt += f"\nTable Purpose: {table_context}"
        
        # Add SQL context if available (with reasonable size limit)
        if sql_context:
            # Extract relevant SQL for this column to provide better context
            # Look for the column name in the SQL
            relevant_sql = ""
            if column_name in sql_context:
                # Try to find SQL snippets related to this column
                lines = sql_context.split("\n")
                for i, line in enumerate(lines):
                    if column_name in line:
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        relevant_sql += "\n".join(lines[start:end]) + "\n\n"
            
            # If no relevant SQL found or it's too short, use a portion of the full SQL
            if len(relevant_sql) < 100 and sql_context:
                relevant_sql = sql_context[:1200] + "..." if len(sql_context) > 1200 else sql_context
            
            prompt += f"\n\nSQL Context: {relevant_sql}"
        
        prompt += """
        
        Base your description on the naming conventions, data type, and context provided. Be specific about:
        1. What data this column contains
        2. The purpose of this data in the model
        3. Any business meaning or calculation logic if apparent
        
        Keep the description informative and useful for data analysts.
        """
        
        # Create specific payload for column descriptions
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 300,  # Adequate tokens for column descriptions
            }
        }
        
        try:
            # Make the API request
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error from Gemini API: {response.status_code} - {response.text}")
                return None
            
            # Parse the response
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        description = parts[0]["text"].strip()
                        if description.endswith('...') or description.endswith('…'):
                            print(f"Warning: Column description appears to be truncated: {description}")
                        return description
            
            print("Unexpected response format from Gemini API for column description")
            return None
            
        except Exception as e:
            print(f"Error calling Gemini API for column description: {str(e)}")
            return None
    
    def generate_model_description(self, model_name: str, project_name: str, 
                                 sql_code: str = None, column_info: List[Dict[str, Any]] = None) -> Optional[str]:
        """Generate a description for a model based on its name, SQL code, and column information"""
        # Start building the prompt with the model name and project
        prompt = f"""
        As a dbt expert, provide a concise and accurate description (3-5 sentences) for a dbt model.

        Model Name: {model_name}
        Project: {project_name}
        """
        
        # Add column information if available
        if column_info and len(column_info) > 0:
            prompt += f"\n\nColumns ({len(column_info)}):\n"
            for column in column_info[:15]:  # Include more columns for better context
                prompt += f"- {column.get('name', 'Unknown')} ({column.get('type', 'Unknown')})\n"
            
            if len(column_info) > 15:
                prompt += f"- ... and {len(column_info) - 15} more columns\n"
        
        # Add SQL code context if available (increased size for better context)
        if sql_code:
            sql_excerpt = sql_code[:1500] + "..." if len(sql_code) > 1500 else sql_code
            prompt += f"\n\nSQL Code:\n{sql_excerpt}"
        
        prompt += """
        
        Based on the model name, columns, and SQL code:
        1. Describe the purpose of this model
        2. Explain what data it processes or produces
        3. Mention its role in the data pipeline
        4. Note any important transformations or business logic
        
        Keep the description clear, technical, and useful for data analysts. Provide a complete description without any truncation.
        """
        
        # Make the API request with increased token limit for model descriptions
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 400,  # Increased token limit for model descriptions
            }
        }
        
        try:
            # Make the API request
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error from Gemini API: {response.status_code} - {response.text}")
                return None
            
            # Parse the response
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
            
            print("Unexpected response format from Gemini API for model description")
            return None
            
        except Exception as e:
            print(f"Error calling Gemini API for model description: {str(e)}")
            return None
    
    def enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with AI-generated descriptions"""
        # Don't proceed if AI descriptions are disabled or the service is not available
        if not self.api_key:
            print("Skipping AI enrichment: No API key available")
            return metadata
        
        print("Starting AI-based metadata enrichment...")
        
        # Make a deep copy of the metadata to avoid modifying the original
        enriched_metadata = json.loads(json.dumps(metadata))
        
        # Get models list
        models = enriched_metadata.get("models", [])
        
        # Process models (limit to 5 for testing/dev)
        for i, model in enumerate(models[:5]):
            model_name = model.get("name", "Unknown")
            project_name = model.get("project", "Unknown")
            
            # Skip models that already have descriptions unless forced
            if model.get("description") and not model.get("refresh_description"):
                continue
            
            print(f"Generating description for model {i+1}/5: {model_name}")
            
            # Generate model description
            model_desc = self.generate_model_description(
                model_name,
                project_name,
                model.get("sql", ""),
                model.get("columns", [])
            )
            
            if model_desc:
                # Store AI description
                model["ai_description"] = model_desc
                
                # If no description exists at all, use the AI one as default
                if not model.get("description"):
                    model["description"] = model_desc
                    model["user_edited"] = False
                
                # Clear refresh flag if it exists
                if "refresh_description" in model:
                    del model["refresh_description"]
        
        # Return the enriched metadata
        return enriched_metadata 