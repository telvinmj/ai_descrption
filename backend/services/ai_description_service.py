import os
import json
import requests
import re
import datetime
import time
from typing import Dict, List, Any, Optional, Tuple

class AIDescriptionService:
    """Service for generating AI descriptions for models and columns using Gemini API"""
    
    def __init__(self, gemini_api_key=None, debug_mode=False):
        """
        Initialize AI Description Service.
        
        Args:
            gemini_api_key: API key for Gemini API (optional)
            debug_mode: Enable additional debug output
        """
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
            print("Set it using: export GEMINI_API_KEY='your-api-key'")
        else:
            print(f"Using Gemini API key: {self.gemini_api_key[:5]}...{self.gemini_api_key[-4:]}")
        
        # Updated to use gemini-2.0-flash-lite model as specified
        self.api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-lite:generateContent"
    
        # Initialize domain knowledge cache
        self.domain_knowledge = {}
        self.column_patterns = self._initialize_column_patterns()
        self.uncertain_descriptions = set()
        
        # Track additional prompt context that can be added to all prompts
        self.additional_prompt_context = ""
        
        # Debugging flags
        self.debug_mode = debug_mode
        print(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'} (set DEBUG_MODE=false to disable)")
        
        # Model availability flag
        self.model_available = bool(self.gemini_api_key)

    def _initialize_column_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common patterns for column naming conventions"""
        return {
            # Time-related columns
            "created_at": {"domain": "timestamp", "description": "The date and time when the record was created"},
            "updated_at": {"domain": "timestamp", "description": "The date and time when the record was last updated"},
            "deleted_at": {"domain": "timestamp", "description": "The date and time when the record was soft-deleted, if applicable"},
            
            # Identification columns
            "_id$": {"domain": "identifier", "description": "Unique identifier for the {entity} record"},
            "^id$": {"domain": "identifier", "description": "Primary key that uniquely identifies this record"},
            "^pk_": {"domain": "identifier", "description": "Primary key that uniquely identifies this record"},
            "^fk_": {"domain": "foreign_key", "description": "Foreign key reference to the {reference_table} table"},
            
            # Amount/metric columns
            "_amount$": {"domain": "financial", "description": "The monetary amount of {entity_context}"},
            "_total$": {"domain": "metric", "description": "The total {entity_context} value"},
            "_count$": {"domain": "metric", "description": "The count of {entity_context}"},
            
            # Status columns
            "_status$": {"domain": "status", "description": "The current status of the {entity}"},
            "is_": {"domain": "boolean", "description": "Flag indicating whether the {entity} {context}"},
            "has_": {"domain": "boolean", "description": "Flag indicating whether the {entity} has {context}"},
            
            # Common business domains
            "customer_": {"domain": "customer", "description": "Customer-related {context}"},
            "product_": {"domain": "product", "description": "Product-related {context}"},
            "order_": {"domain": "order", "description": "Order-related {context}"},
            
            # Date dimensions
            "year": {"domain": "date_dimension", "description": "The year component of the date"},
            "month": {"domain": "date_dimension", "description": "The month component of the date"},
            "day": {"domain": "date_dimension", "description": "The day component of the date"},
            "quarter": {"domain": "date_dimension", "description": "The quarter (Q1-Q4) of the year"}
        }
    
    def _make_api_request(self, prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> Optional[str]:
        """Make a request to Gemini API to generate content"""
        if not self.gemini_api_key:
            print("Error: Gemini API key not set")
            return None
        
        try:
            if self.debug_mode:
                print("\n=== Sending request to Gemini API ===")
                print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
                print(f"Prompt length: {len(prompt)} characters")
                # Print first and last 100 chars of prompt for context
                print(f"Prompt preview: {prompt[:100]}... ... {prompt[-100:] if len(prompt) > 100 else ''}")
            else:
                print(f"Sending API request (max_tokens={max_tokens}, temp={temperature})...")
            
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
                    "temperature": temperature,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": max_tokens,
                }
            }
            
            # Make the API request
            response = requests.post(
                f"{self.api_url}?key={self.gemini_api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error from Gemini API: {response.status_code} - {response.text}")
                return None
            
            if self.debug_mode:
                print("\n=== Received response from Gemini API ===")
                print(f"Response status code: {response.status_code}")
            else:
                print(f"Received API response (status code: {response.status_code})")
            
            # Parse the response
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        # Get the full text without any truncation
                        description = parts[0]["text"].strip()
                        
                        # Print the raw API response for debugging
                        if self.debug_mode:
                            print("\n=== Raw Gemini API Response ===")
                            print(f"Response length: {len(description)} characters")
                            print("Response content:")
                            print("---BEGIN GEMINI RESPONSE---")
                            print(description)
                            print("---END GEMINI RESPONSE---\n")
                        else:
                            print(f"Received text response ({len(description)} chars)")
                        
                        # Check for truncation indicators and log a warning if found
                        if description.endswith('...') or description.endswith('…'):
                            print(f"Warning: AI description appears to be truncated: {description}")
                        return description
            
            print("Unexpected response format from Gemini API")
            if self.debug_mode:
                print(f"Raw response: {response.text[:500]}...")
            return None
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None

    def _extract_domain_from_manifest(self, manifest_data: Dict[str, Any], column_name: str, model_name: str) -> Dict[str, Any]:
        """Extract domain-specific context from manifest data for a column"""
        domain_context = {}
        
        # Look for the model in manifest
        if not manifest_data or "nodes" not in manifest_data:
            return domain_context
        
        # Find the model node
        model_node = None
        for node_id, node in manifest_data.get("nodes", {}).items():
            if node.get("resource_type") == "model" and node.get("name") == model_name:
                model_node = node
                break
        
        if not model_node:
            return domain_context
        
        # Extract column definitions if available
        columns_metadata = model_node.get("columns", {})
        if column_name in columns_metadata:
            column_meta = columns_metadata[column_name]
            if "description" in column_meta:
                domain_context["existing_description"] = column_meta["description"]
            if "tests" in column_meta:
                domain_context["tests"] = column_meta["tests"]
            if "meta" in column_meta:
                domain_context["meta"] = column_meta["meta"]
        
        # Extract tags at model level
        if "tags" in model_node:
            domain_context["model_tags"] = model_node["tags"]
        
        # Extract model description
        if "description" in model_node:
            domain_context["model_description"] = model_node["description"]
        
        # Extract model metadata
        if "meta" in model_node:
            domain_context["model_meta"] = model_node["meta"]
        
        # Look for column references in SQL
        if "compiled_sql" in model_node or "raw_sql" in model_node:
            sql = model_node.get("compiled_sql", model_node.get("raw_sql", ""))
            # Find column references and their context
            domain_context["sql_references"] = self._find_column_references(sql, column_name)
        
        return domain_context

    def _find_column_references(self, sql: str, column_name: str) -> List[str]:
        """Find context around column references in SQL"""
        references = []
        if not sql or not column_name:
            return references
            
        # Find lines containing the column name
        lines = sql.split("\n")
        for i, line in enumerate(lines):
            if column_name in line:
                # Get context (2 lines before and after)
                start = max(0, i-2)
                end = min(len(lines), i+3)
                context = "\n".join(lines[start:end])
                references.append(context)
                
        return references[:5]  # Limit to 5 references for manageability

    def _match_column_pattern(self, column_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Match column name against known patterns to infer domain and description"""
        for pattern, info in self.column_patterns.items():
            # Check if pattern is in column name or matches regex
            if pattern in column_name or re.search(pattern, column_name):
                # Create a copy of the info to customize
                result = info.copy()
                
                # Replace placeholders with context
                if "{entity}" in result.get("description", ""):
                    # Extract entity from model name (users_fact -> user)
                    entity = model_name.split('_')[0].rstrip('s')
                    result["description"] = result["description"].replace("{entity}", entity)
                    
                if "{reference_table}" in result.get("description", ""):
                    # For foreign keys, try to extract referenced table
                    match = re.search(r"fk_(\w+)_id", column_name)
                    ref_table = match.group(1) if match else "referenced"
                    result["description"] = result["description"].replace("{reference_table}", ref_table)
                
                # Replace any remaining placeholders with parts of the column name
                if "{entity_context}" in result.get("description", ""):
                    context = column_name.replace("_amount", "").replace("_total", "").replace("_count", "")
                    result["description"] = result["description"].replace("{entity_context}", context)
                
                if "{context}" in result.get("description", ""):
                    # Extract context from column name after the pattern
                    context = column_name.replace(pattern.replace("^", "").replace("$", ""), "")
                    result["description"] = result["description"].replace("{context}", context)
                
                return result
        
        return None

    def generate_column_description(self, column_name: str, model_name: str, 
                                   sql_context: str = None, column_type: str = None, 
                                   table_context: str = None, manifest_data: Dict[str, Any] = None,
                                   sample_data: List[Any] = None) -> Tuple[str, bool]:
        """
        Generate a domain-aware description for a column based on its name, context, and manifest data.
        
        Returns:
            Tuple[str, bool]: (description, is_confident) - The generated description and confidence flag
        """
        # Check if we have a cached description for this column in this model
        cache_key = f"{model_name}.{column_name}"
        if cache_key in self.domain_knowledge:
            return self.domain_knowledge[cache_key], True
        
        # Check for pattern-based descriptions first
        pattern_match = self._match_column_pattern(column_name, model_name)
        
        # Extract domain-specific context if manifest data is available
        domain_context = {}
        if manifest_data:
            domain_context = self._extract_domain_from_manifest(manifest_data, column_name, model_name)
            
        # If we found a pattern match and there's no existing description or SQL context is limited,
        # we can use the pattern-based description with high confidence
        if pattern_match and (not domain_context.get("existing_description") and 
                           (not sql_context or len(sql_context) < 100)):
            description = pattern_match["description"]
            self.domain_knowledge[cache_key] = description
            return description, True
        
        # Build a detailed prompt with all available context
        prompt = f"""
        As a database expert, provide a concise, accurate, and helpful description (1-2 sentences) for a column in a dbt model.

        Column Name: {column_name}
        Model/Table Name: {model_name}
        Column Data Type: {column_type or 'Unknown'}
        """
        
        # Add domain-specific context from manifest
        if domain_context:
            prompt += "\n\n-- Domain Context --\n"
            
            if "existing_description" in domain_context:
                prompt += f"Existing Description: {domain_context['existing_description']}\n"
                
            if "model_description" in domain_context:
                prompt += f"Model Description: {domain_context['model_description']}\n"
                
            if "model_tags" in domain_context:
                prompt += f"Model Tags: {', '.join(domain_context['model_tags'])}\n"
                
            if "tests" in domain_context:
                prompt += f"Column Tests: {', '.join(domain_context['tests'])}\n"
                
            if "sql_references" in domain_context and domain_context["sql_references"]:
                prompt += f"\nSQL References to this column:\n"
                for ref in domain_context["sql_references"][:2]:  # Limit to 2 for brevity
                    prompt += f"```\n{ref}\n```\n"
        
        # Add table context if available
        if table_context:
            prompt += f"\nTable Purpose: {table_context}\n"
        
        # Add sample data if available
        if sample_data:
            prompt += f"\nSample Values: {', '.join(str(x) for x in sample_data[:5])}\n"
        
        # Add SQL context if available (with reasonable size limit)
        if sql_context and not domain_context.get("sql_references"):
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
        
        Base your description on the naming conventions, data type, domain knowledge, and context provided. Be specific about:
        1. What data this column contains
        2. The purpose of this data in the model
        3. Any business meaning or calculation logic if apparent
        
        At the end, add a confidence level between 1-5 where:
        1: Very uncertain, need more information
        2: Somewhat uncertain, educated guess
        3: Moderately confident based on naming and context
        4: Confident based on clear evidence
        5: Very confident with strong supporting information
        
        Format your response as:
        DESCRIPTION: Your description here
        CONFIDENCE: Your confidence level (1-5)
        """
        
        # Create specific payload for column descriptions
        try:
            # Make the API request
            response = self._make_api_request(prompt, max_tokens=800)
            
            if not response:
                # If API fails but we have a pattern match, use that
                if pattern_match:
                    return pattern_match["description"], False
                return "No description available", False
            
            # Extract description and confidence
            desc_match = re.search(r"DESCRIPTION:\s*(.*?)(?:\nCONFIDENCE|\Z)", response, re.DOTALL)
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", response)
            
            description = desc_match.group(1).strip() if desc_match else response.strip()
            confidence = int(conf_match.group(1)) if conf_match else 0
            
            # Clean up the description
            description = description.replace("DESCRIPTION:", "").strip()
            
            # Store in domain knowledge cache
            self.domain_knowledge[cache_key] = description
            
            # If confidence is low, mark this description as uncertain
            is_confident = confidence >= 3
            if not is_confident:
                self.uncertain_descriptions.add(cache_key)
                
            return description, is_confident
            
        except Exception as e:
            print(f"Error generating column description: {str(e)}")
            # Fallback to pattern matching if we have it
            if pattern_match:
                return pattern_match["description"], False
            return "No description available", False
    
    def get_uncertain_descriptions(self) -> List[Dict[str, Any]]:
        """Get a list of columns with uncertain descriptions"""
        uncertain_list = []
        for column_key in self.uncertain_descriptions:
            model_name, column_name = column_key.split(".")
            uncertain_list.append({
                "model_name": model_name,
                "column_name": column_name,
                "current_description": self.domain_knowledge.get(column_key, "")
            })
        return uncertain_list
    
    def improve_description_with_feedback(self, model_name: str, column_name: str, 
                                         feedback: str, sample_data: List[Any] = None) -> str:
        """Improve a column description with user feedback and optional sample data"""
        cache_key = f"{model_name}.{column_name}"
        current_description = self.domain_knowledge.get(cache_key, "No current description")
        
        prompt = f"""
        As a database expert, improve this column description using new information.
        
        Model Name: {model_name}
        Column Name: {column_name}
        Current Description: {current_description}
        
        User Feedback: {feedback}
        """
        
        # Add sample data if provided
        if sample_data:
            prompt += f"\nSample Values: {', '.join(str(x) for x in sample_data[:10])}\n"
        
        prompt += """
        
        Based on this new information, please provide an improved description that is:
        1. Concise (1-2 sentences)
        2. Specific to this column's purpose
        3. Incorporates the user feedback
        4. Technical yet accessible for data analysts
        
        IMPROVED DESCRIPTION:
        """
        
        # Make the API request
        improved_desc = self._make_api_request(prompt, max_tokens=600)
        
        if not improved_desc:
            return current_description
            
        # Clean up the response
        improved_desc = improved_desc.replace("IMPROVED DESCRIPTION:", "").strip()
        
        # Update the cache and remove from uncertain list
        self.domain_knowledge[cache_key] = improved_desc
        if cache_key in self.uncertain_descriptions:
            self.uncertain_descriptions.remove(cache_key)
            
        return improved_desc
    
    def generate_model_description(self, model_name: str, project_name: str, 
                                 sql_code: str = None, column_info: List[Dict[str, Any]] = None,
                                 manifest_data: Dict[str, Any] = None) -> Optional[str]:
        """Generate a description for a model based on its name, SQL code, column information and manifest data"""
        # Start building the prompt with the model name and project
        prompt = f"""
        As a dbt expert, provide a concise and accurate description (2-3 sentences) for a dbt model.

        Model Name: {model_name}
        Project: {project_name}
        """
        
        # Add domain context from manifest if available
        if manifest_data and "nodes" in manifest_data:
            for node_id, node in manifest_data.get("nodes", {}).items():
                if node.get("resource_type") == "model" and node.get("name") == model_name:
                    if "description" in node:
                        prompt += f"\nExisting Description: {node['description']}\n"
                    if "tags" in node:
                        prompt += f"\nTags: {', '.join(node['tags'])}\n"
                    if "meta" in node:
                        prompt += f"\nMetadata: {json.dumps(node['meta'])}\n"
                    break
        
        # Add column information if available
        if column_info and len(column_info) > 0:
            prompt += f"\n\nColumns ({len(column_info)}):\n"
            for column in column_info[:15]:  # Include more columns for better context
                desc = column.get("description", "").strip()
                col_desc = f" - {desc}" if desc else ""
                prompt += f"- {column.get('name', 'Unknown')} ({column.get('type', 'Unknown')}){col_desc}\n"
            
            if len(column_info) > 15:
                prompt += f"- ... and {len(column_info) - 15} more columns\n"
        
        # Add SQL code context if available
        if sql_code:
            sql_excerpt = sql_code[:1500] + "..." if len(sql_code) > 1500 else sql_code
            prompt += f"\n\nSQL Code:\n{sql_excerpt}"
        
        prompt += """
        
        Based on the model name, columns, and SQL code:
        1. Describe the purpose of this model
        2. Explain what data it processes or produces
        3. Mention its role in the data pipeline
        4. Note any important transformations or business logic
        
        Keep the description clear, technical, and useful for data analysts.
        """
        
        # Make the API request
        return self._make_api_request(prompt, max_tokens=800)
    
    def generate_model_with_columns_description(self, model: Dict[str, Any], manifest_data: Dict[str, Any] = None, force_update: bool = False) -> Dict[str, Any]:
        """
        Generate descriptions for a model and all its columns in a single API request.
        This is more efficient than making separate API calls for the model and each column.
        
        Args:
            model: The model dictionary containing all model information
            manifest_data: Optional manifest data for domain context
            force_update: If True, overwrite existing descriptions with AI-generated ones
            
        Returns:
            Dict containing the model with updated descriptions
        """
        import time
        start_time = time.time()
        
        model_name = model.get("name", "Unknown")
        project_name = model.get("project", "Unknown")
        columns = model.get("columns", [])
        sql_code = model.get("sql", "")
        
        print(f"BATCH PROCESSING: Generating descriptions for model {model_name} and {len(columns)} columns in a SINGLE API request")
        print(f"Force update mode: {force_update}")
        
        # Special handling for models with no columns
        if not columns:
            print(f"Model {model_name} has no columns but we'll still generate a model description")
            
            # Check if the model already has a description to preserve (unless force_update is True)
            if (model.get("description") or model.get("ai_description")) and not force_update:
                print(f"Model {model_name} already has descriptions, preserving them.")
                end_time = time.time()
                print(f"Model processing skipped, took {end_time - start_time:.2f} seconds")
                return model
            
            # If force_update is True, we continue to generate a new description even if one exists
            if force_update and (model.get("description") or model.get("ai_description")):
                print(f"Force update enabled: Regenerating description for model {model_name}")
            
            # Create a prompt specifically for model-only description
            prompt = f"""
            You are a data model expert with deep domain knowledge in financial services, insurance, and business data.
            I need you to generate a description for a data model that has no columns defined yet.
            
            MODEL INFORMATION:
            Model Name: {model_name}
            Project: {project_name}
            """
            
            # Add domain context from manifest if available
            domain_context = {}
            if manifest_data:
                for node_id, node in manifest_data.get("nodes", {}).items():
                    if node.get("resource_type") == "model" and node.get("name") == model_name:
                        if "description" in node:
                            domain_context["model_description"] = node["description"]
                        if "tags" in node:
                            domain_context["model_tags"] = node["tags"]
                        if "meta" in node:
                            domain_context["model_meta"] = node["meta"]
                        break
                        
            if domain_context:
                prompt += "\n--- Domain Context ---\n"
                if "model_description" in domain_context:
                    prompt += f"Existing Model Description: {domain_context['model_description']}\n"
                if "model_tags" in domain_context:
                    prompt += f"Model Tags: {', '.join(domain_context['model_tags'])}\n"
                if "model_meta" in domain_context:
                    prompt += f"Model Metadata: {json.dumps(domain_context['model_meta'])}\n"
            
            # Add SQL code if available (often contains clues about the model purpose)
            if sql_code:
                sql_excerpt = sql_code[:1500] + "..." if len(sql_code) > 1500 else sql_code
                prompt += f"\n--- SQL Code ---\n{sql_excerpt}\n"
            
            prompt += """
            
            TASK:
            Based on the model name and any other context provided, generate a comprehensive description
            of what this model likely represents and its purpose in the data warehouse.
            
            Your description should:
            1. Start with "This model represents..." or "This table contains..."
            2. Explain what business entity or process this model likely represents
            3. Describe what the data is probably used for
            4. Be specific about its likely role in the business context
            5. Be 2-3 sentences in length
            
            Return ONLY the description text with no additional formatting or commentary.
            """
            
            # Make the API request
            print(f"Sending API request for model-only description of {model_name}...")
            api_start_time = time.time()
            response = self._make_api_request(prompt, max_tokens=800, temperature=0.1)
            api_end_time = time.time()
            print(f"API request for model-only description completed in {api_end_time - api_start_time:.2f} seconds")
            
            if response:
                # Store the response directly as the model description
                print('****************************')
                print(response)
                print('****************************')
                
                model["ai_description"] = response.strip()
                model["description"] = response.strip()
                model["user_edited"] = False
                print(f"  - Added description for model {model_name}")
            
            end_time = time.time()
            print(f"Model-only description generation completed in {end_time - start_time:.2f} seconds")
            return model
        
        # Continue with regular processing for models with columns
        print(f"This replaces {len(columns) + 1} individual API calls with just 1 call, significantly improving efficiency")
        
        # Extract domain-specific context if manifest data is available
        domain_context = {}
        if manifest_data:
            # Find the model node in manifest
            model_node = None
            for node_id, node in manifest_data.get("nodes", {}).items():
                if node.get("resource_type") == "model" and node.get("name") == model_name:
                    model_node = node
                    break
            
            if model_node:
                # Extract model-level context
                if "description" in model_node:
                    domain_context["model_description"] = model_node["description"]
                if "tags" in model_node:
                    domain_context["model_tags"] = model_node["tags"]
                if "meta" in model_node:
                    domain_context["model_meta"] = model_node["meta"]
        
        # Determine the model type based on naming convention
        model_type = "data model"
        if model_name.startswith("dim_"):
            model_type = "dimension table"
        elif model_name.startswith("fct_"):
            model_type = "fact table"
        
        # Build a detailed prompt that includes the model and all its columns
        prompt = f"""
        You are a data model expert with deep domain knowledge in financial services, insurance, and business data. 
        I need you to analyze a data model and its columns, providing detailed descriptions.
        
        MODEL INFORMATION:
        Model Name: {model_name}
        Project: {project_name}
        Type: {model_type}
        
        CONTEXT FOR COLUMN NAMES:
        - Column names in this organization are often abbreviated or use technical shorthand
        - Use the column name structure, data type, and model context to determine the business meaning
        - Look for common patterns like id/cd/dt/nm suffixes which often mean identifier/code/date/name
        - Column prefixes often indicate the subject domain (e.g., cust_ for customer)
        - Use SQL context to help understand how columns are populated and what they represent
        - Pay attention to patterns across different columns - they often follow similar conventions
        """
        
        # Add additional prompt context if available
        if self.additional_prompt_context:
            prompt += f"""
            
            ADDITIONAL DOMAIN CONTEXT:
            {self.additional_prompt_context}
        """
        
        # Add domain context if available
        if domain_context:
            prompt += "\n--- Domain Context ---\n"
            if "model_description" in domain_context:
                prompt += f"Existing Model Description: {domain_context['model_description']}\n"
            if "model_tags" in domain_context:
                prompt += f"Model Tags: {', '.join(domain_context['model_tags'])}\n"
        
        # Add all columns with their types
        prompt += "\n--- Columns ---\n"
        # Process columns in a single batch, but respect token limits
        max_columns_per_request = 200  # Maximum columns per API request with 16000 token limit
        columns_to_process = columns[:max_columns_per_request]
        
        for i, column in enumerate(columns_to_process):
            col_name = column.get("name", "")
            col_type = column.get("type", "unknown")
            prompt += f"{i+1}. {col_name} ({col_type})\n"
        
        # If there are more columns than we're processing, note that
        if len(columns) > max_columns_per_request:
            prompt += f"... and {len(columns) - max_columns_per_request} more columns (not included due to size constraints)\n"
        
        # Add SQL code context (trimmed to avoid token limits)
        if sql_code:
            sql_excerpt = sql_code[:1500] + "..." if len(sql_code) > 1500 else sql_code
            prompt += f"\n--- SQL Code ---\n{sql_excerpt}\n"
        
        # Provide clear instructions for the format of the response
        prompt += """
        TASK:
        1. First, analyze the model as a whole to understand its business purpose.
        2. Then, analyze each column to determine its true business meaning, even if it has an abbreviated or technical name.
        3. Return structured descriptions of the model and all its columns.
        
        INSTRUCTIONS FOR QUALITY DESCRIPTIONS:
        
        For the MODEL description:
        - Start with "This model represents..." or "This table contains..."
        - Explain what business entity or process this model represents
        - Describe what the data is used for and who might use it
        - Be specific about its role in the business context
        - 2-3 sentences in length
        
        For COLUMN descriptions:
        - Start with "This column stores..." or "This field represents..."
        - Decode any abbreviations in the column name
        - Explain the business purpose, not just the technical meaning
        - For each column, consider:
          * What real-world attribute or measure does it represent?
          * Why would a business user need this information?
          * How does it relate to the overall model purpose?
        - 1-2 sentences per column
        
        Respond ONLY in this JSON format:
        
        {
          "model_description": "Your comprehensive model description here",
          "model_confidence": 5,
          "columns": [
            {
              "name": "column_name1",
              "description": "Business-focused column description",
              "confidence": 5
            },
            {
              "name": "column_name2",
              "description": "Business-focused column description",
              "confidence": 4
            }
            ... etc for all columns
          ]
        }
        
        Confidence levels:
        - 5: Clear understanding with high certainty
        - 4: Good understanding with reasonable certainty
        - 3: Moderate understanding with some uncertainty
        - 2: Limited understanding with significant uncertainty
        - 1: Minimal understanding, highly uncertain
        """
        
        # Make the API request with increased token limit
        print(f"Sending a single batch API request for model {model_name} with {len(columns_to_process)} columns...")
        api_start_time = time.time()
        response = self._make_api_request(prompt, max_tokens=16000, temperature=0.1)  # Increased from 8000 to 16000 for handling up to 200 columns
        api_end_time = time.time()
        print(f"API request completed in {api_end_time - api_start_time:.2f} seconds")
        
        if not response:
            print(f"Failed to generate descriptions for model {model_name}")
            end_time = time.time()
            print(f"Model processing failed, took {end_time - start_time:.2f} seconds")
            return model  # Return unchanged model if API request failed
        
        try:
            # Extract JSON response - handle potential formatting issues
            json_str = response.strip()
            
            # Remove any leading/trailing text that isn't part of the JSON
            json_start = json_str.find('{')
            json_end = json_str.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = json_str[json_start:json_end]
            
            # Print the raw response for debugging
            if self.debug_mode:
                print(f"DEBUG - Raw JSON response for model {model_name}:")
                print(f"Response length: {len(json_str)} characters")
                print(f"JSON START: {json_str[:200]}...")
                print(f"JSON END: ...{json_str[-200:]}")
                print("\nParsing JSON response...")
            
            # Parse the JSON response
            descriptions = json.loads(json_str)
            
            # Print full parsed JSON response for debugging purposes
            if self.debug_mode:
                print("\n=== PARSED JSON RESPONSE ===")
                print(json.dumps(descriptions, indent=2))
                print("=== END PARSED JSON RESPONSE ===\n")
                
            print(f"Successfully received batch response with descriptions for model and {len(descriptions.get('columns', []))} columns")
            
            # Update model description if provided
            if "model_description" in descriptions and descriptions["model_description"]:
                print(f"DEBUG - Assigning model description for {model_name}:")
            if "model_description" in descriptions and descriptions["model_description"]:
                print(f"DEBUG - Assigning model description for {model_name}:")
                print(f"AI description: {descriptions['model_description'][:100]}...")
                
                model["ai_description"] = descriptions["model_description"]
                
                # If no description exists, use the AI one
                if not model.get("description"):
                    print(f"DEBUG - No existing description found, using AI description as primary description")
                    model["description"] = descriptions["model_description"]
                    model["user_edited"] = False
                    print(f"  - Added AI description for model {model_name} from batch response")
                else:
                    print(f"DEBUG - Existing description found, keeping it and storing AI version separately")
            else:
                print(f"DEBUG - No model description found in API response")
            
            # Extract and store model confidence if available
            if "model_confidence" in descriptions:
                confidence = descriptions["model_confidence"]
                confidence = int(confidence) if isinstance(confidence, (int, str)) else 0
                model["ai_confidence_score"] = confidence
                model["needs_review"] = confidence <= 3
                print(f"DEBUG - Model confidence score: {confidence}")
            
            # Update column descriptions
            if "columns" in descriptions and isinstance(descriptions["columns"], list):
                column_descriptions = {col["name"]: col for col in descriptions["columns"]}
                
                # Print received column descriptions for debugging
                print(f"DEBUG - Received descriptions for {len(column_descriptions)} columns:")
                for col_name, col_data in list(column_descriptions.items())[:2]:  # Show first 2 for debugging
                    print(f"  - Column: {col_name}")
                    print(f"    Description: {col_data.get('description', 'None')[:50]}...")
                    print(f"    Confidence: {col_data.get('confidence', 'Not provided')}")
                
                if len(column_descriptions) > 2:
                    print(f"  - ... and {len(column_descriptions) - 2} more columns")
                
                # Check for missing columns
                model_column_names = [col.get("name") for col in columns_to_process if col.get("name")]
                missing_columns = set(model_column_names) - set(column_descriptions.keys())
                if missing_columns:
                    print(f"WARNING - {len(missing_columns)} columns are missing from API response:")
                    print(f"  Missing columns: {', '.join(list(missing_columns)[:5])}" + 
                          (f"... and {len(missing_columns)-5} more" if len(missing_columns) > 5 else ""))
                    
                    # For missing columns, we'll request supplementary descriptions in a separate request
                    if missing_columns:
                        print(f"Generating supplementary descriptions for {len(missing_columns)} missing columns...")
                        missing_col_list = list(missing_columns)
                        missing_cols_to_process = [col for col in columns_to_process if col.get("name") in missing_col_list]
                        
                        if missing_cols_to_process:
                            # Create a supplementary prompt for just the missing columns
                            supp_prompt = f"""
                            You are a data expert. Please provide descriptions for these columns from the model {model_name}:
                            
                            """
                            for i, col in enumerate(missing_cols_to_process):
                                col_name = col.get("name", "")
                                col_type = col.get("type", "unknown")
                                supp_prompt += f"{i+1}. {col_name} ({col_type})\n"
                                
                            supp_prompt += """
                            Analyze each column name carefully to determine its business meaning.
                            Respond with a JSON object containing descriptions for each column:
                            
                            {
                              "columns": [
                                {
                                  "name": "column_name",
                                  "description": "Clear business description of what this column represents",
                                  "confidence": 1-5 (where 5 is highest confidence)
                                },
                                ...more columns...
                              ]
                            }
                            
                            For confidence scores, use the following guidelines:
                            5 = Very high confidence (Almost certain)
                            4 = High confidence (Strong certainty)
                            3 = Moderate confidence (Reasonable guess)
                            2 = Low confidence (Uncertain)
                            1 = Very low confidence (Highly uncertain)
                            """
                            
                            # Make supplementary API request
                            supp_response = self._make_api_request(supp_prompt, max_tokens=1600, temperature=0.1)
                            
                            if supp_response:
                                try:
                                    # Extract and parse JSON
                                    supp_json_str = supp_response.strip()
                                    supp_json_start = supp_json_str.find('{')
                                    supp_json_end = supp_json_str.rfind('}') + 1
                                    if supp_json_start >= 0 and supp_json_end > supp_json_start:
                                        supp_json_str = supp_json_str[supp_json_start:supp_json_end]
                                        
                                    supp_descriptions = json.loads(supp_json_str)
                                    
                                    if "columns" in supp_descriptions and isinstance(supp_descriptions["columns"], list):
                                        for col_data in supp_descriptions["columns"]:
                                            if "name" in col_data and "description" in col_data:
                                                column_descriptions[col_data["name"]] = col_data
                                                print(f"  - Added supplementary description for {col_data['name']}")
                                except:
                                    print("Error processing supplementary descriptions")
                
                columns_updated = 0
                for column in model.get("columns", []):
                    col_name = column.get("name")
                    if col_name and col_name in column_descriptions:
                        desc_data = column_descriptions[col_name]
                        
                        # Store AI description
                        column["ai_description"] = desc_data.get("description", "")
                        
                        # Update the description field based on the force_update flag and user_edited status
                        if force_update:
                            if not column.get("user_edited", False):
                                column["description"] = desc_data.get("description", "")
                                columns_updated += 1
                                print(f"DEBUG - Force updated description for column {col_name}")
                            else:
                                print(f"DEBUG - Column {col_name} is user-edited, preserving user's description even with force_update")
                        # Standard behavior (no force update)
                        elif not column.get("description"):
                            column["description"] = desc_data.get("description", "")
                            columns_updated += 1
                            print(f"DEBUG - Added new description for empty column {col_name}")
                        else:
                            print(f"DEBUG - Column {col_name} already has description, storing AI version separately")
                        
                        # Store confidence level
                        confidence = desc_data.get("confidence", 0)
                        confidence = int(confidence) if isinstance(confidence, (int, str)) else 0
                        column["ai_confidence_score"] = confidence
                        column["needs_review"] = confidence <= 3
                        
                        # Store description in domain knowledge cache
                        cache_key = f"{model_name}.{col_name}"
                        self.domain_knowledge[cache_key] = desc_data.get("description", "")
                        if confidence <= 3 and desc_data.get("description"):
                            self.uncertain_descriptions.add(cache_key)
                
                print(f"  - Updated {columns_updated} column descriptions from batch response")
            else:
                print(f"DEBUG - No column descriptions found in API response or invalid format")
                print(f"Response keys: {list(descriptions.keys())}")
            
            # Process columns not included in the API request (if any) with a dynamic batch approach
            remaining_columns = columns[max_columns_per_request:] if len(columns) > max_columns_per_request else []
            if remaining_columns:
                print(f"Processing {len(remaining_columns)} additional columns not included in the initial API request...")
                
                # Process remaining columns in batches of 200
                for i in range(0, len(remaining_columns), max_columns_per_request):
                    batch = remaining_columns[i:i + max_columns_per_request]
                    print(f"Processing batch of {len(batch)} additional columns...")
                    
                    # Create a new prompt just for these columns
                    batch_prompt = f"""
                    You are a data model expert. Please provide descriptions for these additional columns from the model {model_name}:
                    
                    """
                    for j, col in enumerate(batch):
                        col_name = col.get("name", "")
                        col_type = col.get("type", "unknown")
                        batch_prompt += f"{j+1}. {col_name} ({col_type})\n"
                        
                    batch_prompt += """
                    Analyze each column name carefully to determine its business meaning.
                    Respond with a JSON object containing descriptions for each column:
                    
                    {
                      "columns": [
                        {
                          "name": "column_name",
                          "description": "Clear business description of what this column represents",
                          "confidence": 1-5 (where 5 is highest confidence)
                        },
                        ...more columns...
                      ]
                    }
                    
                    For confidence scores, use the following guidelines:
                    5 = Very high confidence (Almost certain)
                    4 = High confidence (Strong certainty)
                    3 = Moderate confidence (Reasonable guess)
                    2 = Low confidence (Uncertain)
                    1 = Very low confidence (Highly uncertain)
                    """
                    
                    # Make batch API request
                    batch_response = self._make_api_request(batch_prompt, max_tokens=8000, temperature=0.1)
                    
                    if batch_response:
                        try:
                            # Extract and parse JSON
                            batch_json_str = batch_response.strip()
                            batch_json_start = batch_json_str.find('{')
                            batch_json_end = batch_json_str.rfind('}') + 1
                            if batch_json_start >= 0 and batch_json_end > batch_json_start:
                                batch_json_str = batch_json_str[batch_json_start:batch_json_end]
                                
                            batch_descriptions = json.loads(batch_json_str)
                            
                            if "columns" in batch_descriptions and isinstance(batch_descriptions["columns"], list):
                                batch_columns_updated = 0
                                
                                # Update column descriptions from batch response
                                for col_data in batch_descriptions["columns"]:
                                    if "name" in col_data and "description" in col_data:
                                        col_name = col_data["name"]
                                        # Find the column in the model
                                        for column in remaining_columns:
                                            if column.get("name") == col_name:
                                                # Store AI description
                                                column["ai_description"] = col_data.get("description", "")
                                                
                                                # If no description exists, use the AI one
                                                if not column.get("description"):
                                                    column["description"] = col_data.get("description", "")
                                                    column["user_edited"] = False
                                                    batch_columns_updated += 1
                                                
                                                # Store confidence level
                                                confidence = col_data.get("confidence", 0)
                                                confidence = int(confidence) if isinstance(confidence, (int, str)) else 0
                                                column["ai_confidence_score"] = confidence
                                                column["needs_review"] = confidence <= 3
                                                
                                                # Store description in domain knowledge cache
                                                cache_key = f"{model_name}.{col_name}"
                                                self.domain_knowledge[cache_key] = col_data.get("description", "")
                                                if confidence <= 3 and col_data.get("description"):
                                                    self.uncertain_descriptions.add(cache_key)
                                                    
                                                break
                                
                                print(f"  - Updated {batch_columns_updated} additional column descriptions")
                                total_columns_updated += batch_columns_updated
                        except Exception as e:
                            print(f"Error processing batch descriptions: {str(e)}")
            
            print(f"BATCH PROCESSING COMPLETE: Finished processing model {model_name} and its columns from a single API request")
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total processing time for model {model_name}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            return model
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response for model {model_name}")
            print(f"Raw response: {response[:500]}...")  # Print first 500 chars for debugging
            end_time = time.time()
            print(f"Model processing failed with JSON error, took {end_time - start_time:.2f} seconds")
            return model
        except Exception as e:
            print(f"Error processing descriptions for model {model_name}: {str(e)}")
            end_time = time.time()
            print(f"Model processing failed with error, took {end_time - start_time:.2f} seconds")
            return model
    
    def batch_process_models(self, models_batch: List[Dict[str, Any]], manifest_data: Dict[str, Any] = None, all_models: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of models to generate descriptions for the models and their columns.
        This batched approach is significantly more efficient than individual processing.
        
        Args:
            models_batch: List of model dictionaries to process
            manifest_data: Optional manifest data for domain context
            all_models: Optional list of all models, needed when processing missing descriptions
            
        Returns:
            List of models with updated descriptions
        """
        import time
        import datetime
        
        if not models_batch:
            return models_batch
            
        # Use models_batch as the reference to all models if all_models is not provided
        models = all_models if all_models is not None else models_batch
        
        # Track timing
        batch_start_time = time.time()
        
        print(f"\n=== BATCH PROCESSING: Processing {len(models_batch)} models in a single API request ===")
        
        # Create list of model information for the prompt
        prompt = "You are a database expert specializing in documenting data models. I need descriptions for the following models:\n\n"
        
        # Add context for additional prompt
        if self.additional_prompt_context:
            prompt += f"ADDITIONAL CONTEXT PROVIDED BY USER:\n{self.additional_prompt_context}\n\n"
            prompt += "Use the above context to improve the relevance and accuracy of your descriptions.\n\n"
        
        # Add each model to the prompt
        for i, model in enumerate(models_batch):
            model_name = model.get("name", "Unknown")
            project_name = model.get("project", "Unknown")
            sql_code = model.get("sql", "")
            columns = model.get("columns", [])
            
            prompt += f"\n---MODEL #{i+1}---\n"
            prompt += f"Model Name: {model_name}\n"
            prompt += f"Project: {project_name}\n"
            
            # Extract model type hint from name
            model_type = "data model"
            if model_name.startswith("dim_"):
                prompt += "Type: Dimension table\n"
            elif model_name.startswith("fct_"):
                prompt += "Type: Fact table\n"
            elif model_name.startswith("raw_"):
                prompt += "Type: Raw source data\n"
            elif model_name.startswith("stg_"):
                prompt += "Type: Staging table\n"
            
            # Add column information (but limit to 200 columns per model)
            max_columns_per_model = 200
            columns_to_process = columns[:max_columns_per_model]
            if columns_to_process:
                prompt += f"\nColumns ({len(columns_to_process)}):\n"
                for column in columns_to_process:
                    col_name = column.get("name", "Unknown")
                    col_type = column.get("type", "Unknown")
                    prompt += f"- {col_name} ({col_type})\n"
            else:
                prompt += "\nThis model has no columns defined.\n"
                
            # Add SQL code context (limited to 1000 chars to avoid token limits)
            if sql_code:
                sql_excerpt = sql_code[:1000] + "..." if len(sql_code) > 1000 else sql_code
                prompt += f"\nSQL Code snippet:\n{sql_excerpt}\n"
                
            # Add domain context from manifest data if available
            if manifest_data:
                domain_info = ""
                for node_id, node in manifest_data.get("nodes", {}).items():
                    if node.get("resource_type") == "model" and node.get("name") == model_name:
                        if "description" in node:
                            domain_info += f"Existing description: {node['description']}\n"
                        if "tags" in node and node["tags"]:
                            domain_info += f"Tags: {', '.join(node['tags'])}\n"
                if domain_info:
                    prompt += f"\nDomain Context:\n{domain_info}\n"
            
        # Add instructions for the response format with explicit examples and more detailed guidance
        prompt += """
        ---MODEL INFORMATION ENDS---
        
        TASK:
        For each model, provide:
        1. A comprehensive model description (2-3 sentences)
        2. Descriptions for ALL columns provided (1-2 sentences each)
        3. A confidence score (1-5) for each model and column description
        
        RESPONSE FORMAT REQUIREMENTS:
        - You MUST provide descriptions for EVERY SINGLE column listed for each model.
        - Ensure each column description is clear, concise, and describes the business meaning.
        - Do not skip any columns, even if they seem technical or unclear.
        - Make your best inference based on the column name, type, and model context.
        - Format each column description exactly as shown in the example below.
        - Include a confidence score from 1 to 5 for each model and column description, where:
          * 5 = Very high confidence (Almost certain)
          * 4 = High confidence (Strong certainty)
          * 3 = Moderate confidence (Reasonable guess)
          * 2 = Low confidence (Uncertain)
          * 1 = Very low confidence (Highly uncertain)
        
        For each model, your response should be formatted as follows:
        
        MODEL #1 DESCRIPTION:
        This is a comprehensive description of what this model represents and its purpose...
        
        MODEL #1 CONFIDENCE: 4
        
        MODEL #1 COLUMNS:
        - column_name1: This column represents... It is used for... [Confidence: 5]
        - column_name2: This column represents... It contains... [Confidence: 3]
        ... and so on for all columns
        
        MODEL #2 DESCRIPTION:
        ... description for model #2
        
        MODEL #2 CONFIDENCE: 3
        
        MODEL #2 COLUMNS:
        - column_name1: This column represents... It is used for... [Confidence: 4]
        - column_name2: This column represents... It contains... [Confidence: 2]
        ... and so on for all columns
        
        For each description, focus on business meaning and purpose. BE THOROUGH - describe EVERY COLUMN without exception.
        """
        
        # Make the API request with increased token limit for multiple models
        print(f"Sending batch API request for {len(models_batch)} models...")
        api_start_time = time.time()
        response = self._make_api_request(prompt, max_tokens=16000, temperature=0.1)  # Increased token limit from 12000 to 16000 to handle up to 200 columns
        api_end_time = time.time()
        api_time = api_end_time - api_start_time
        print(f"API request completed in {api_time:.2f} seconds ({api_time/60:.2f} minutes)")
        
        if not response:
            print(f"Failed to generate descriptions for batch of {len(models_batch)} models")
            return models_batch  # Return unchanged models if API request failed
        
        # Start timing the response processing
        processing_start_time = time.time()
            
        try:
            # Process the response using regex patterns to extract descriptions
            if self.debug_mode:
                print(f"Received response of length {len(response)}")
                print(f"First 300 chars: {response[:300]}")
                print(f"Last 300 chars: {response[-300:]}")
                
            # Track metrics
            models_updated = 0
            columns_updated = 0
            columns_without_descriptions = 0
            total_columns = 0
            
            # Track models and columns that still need descriptions
            models_missing_descriptions = []
            models_with_missing_column_descriptions = {}
            
            # Process each model
            for i, model in enumerate(models_batch):
                model_name = model.get("name", "Unknown")
                model_columns = model.get("columns", [])
                total_columns += len(model_columns)
                
                # Store API response for debugging
                model["_last_api_response"] = response
                
                # Extract model description using regex
                model_desc_pattern = rf"MODEL #{i+1} DESCRIPTION:\s*(.*?)(?:MODEL #{i+2} DESCRIPTION:|MODEL #{i+1} CONFIDENCE:|MODEL #{i+1} COLUMNS:|$)"
                model_desc_match = re.search(model_desc_pattern, response, re.DOTALL)
                
                model_has_description = False
                if model_desc_match:
                    model_desc = model_desc_match.group(1).strip()
                    model["ai_description"] = model_desc
                    
                    # If no description exists, use the AI one
                    if not model.get("description"):
                        model["description"] = model_desc
                        model["user_edited"] = False
                        models_updated += 1
                        model_has_description = True
                        print(f"  - Added description for model {model_name}")
                else:
                    print(f"  - Warning: Could not find description for model {model_name}")
                    if not model.get("description") and not model.get("ai_description"):
                        models_missing_descriptions.append(model)
                
                # Extract model confidence score
                model_conf_pattern = rf"MODEL #{i+1} CONFIDENCE:\s*(\d+)"
                model_conf_match = re.search(model_conf_pattern, response)
                
                if model_conf_match:
                    confidence = int(model_conf_match.group(1))
                    model["ai_confidence_score"] = confidence
                    print(f"  - Added confidence score {confidence}/5 for model {model_name}")
                else:
                    print(f"  - Warning: Could not find confidence score for model {model_name}")
                
                # Extract column descriptions
                columns_section_pattern = rf"MODEL #{i+1} COLUMNS:(.*?)(?:MODEL #{i+2}|$)"
                columns_section_match = re.search(columns_section_pattern, response, re.DOTALL)
                
                if columns_section_match:
                    columns_text = columns_section_match.group(1).strip()
                    
                    # Process each column
                    model_columns_updated = 0
                    model_columns_without_desc = 0
                    missing_columns_for_model = []
                    
                    for column in model_columns:
                        col_name = column.get("name", "")
                        if not col_name:
                            continue
                            
                        # Find the description for this column
                        col_pattern = rf"- {re.escape(col_name)}:?\s*(.*?)(?:\[Confidence:\s*(\d+)\])?(?:\n- |\n\n|$)"
                        col_match = re.search(col_pattern, columns_text, re.DOTALL)
                        
                        if col_match:
                            col_desc = col_match.group(1).strip()
                            col_conf = col_match.group(2)
                            confidence = int(col_conf) if col_conf else 3
                            
                            # Store AI description
                            column["ai_description"] = col_desc
                            
                            # If no description exists, use the AI one
                            if not column.get("description"):
                                column["description"] = col_desc
                                column["user_edited"] = False
                                model_columns_updated += 1
                            
                            # Store confidence score
                            column["ai_confidence_score"] = confidence
                            column["needs_review"] = confidence <= 3
                        else:
                            # Track columns without descriptions
                            model_columns_without_desc += 1
                            missing_columns_for_model.append(column)
                    
                    # Update counters
                    columns_updated += model_columns_updated
                    columns_without_descriptions += model_columns_without_desc
                    
                    print(f"  - Updated {model_columns_updated} columns for model {model_name}")
                    
                    if model_columns_without_desc > 0:
                        print(f"  - Warning: {model_columns_without_desc} columns still have no description for model {model_name}")
                        # Store models with missing column descriptions
                        if missing_columns_for_model:
                            models_with_missing_column_descriptions[model_name] = {
                                "model": model,
                                "missing_columns": missing_columns_for_model
                            }
                else:
                    print(f"  - Warning: Could not find column descriptions for model {model_name}")
                    # If a model has columns but we couldn't find any descriptions for them
                    if model_columns:
                        models_with_missing_column_descriptions[model_name] = {
                            "model": model,
                            "missing_columns": model_columns
                        }
            
            # Process models with missing descriptions
            if models_missing_descriptions:
                print(f"\n=== PROCESSING {len(models_missing_descriptions)} MODELS WITHOUT DESCRIPTIONS ===")
                
                # Instead of processing one by one, we'll create a new batch prompt for all missing models
                missing_models_prompt = "You are a data model expert. I need descriptions for the following models:\n\n"
                
                # Add each missing model to the prompt
                for i, missing_model in enumerate(models_missing_descriptions):
                    model_name = missing_model.get("name", "Unknown")
                    print(f"  - Including model in batch: {model_name}")
                    
                    missing_models_prompt += f"\n---MODEL #{i+1}---\n"
                    missing_models_prompt += f"Model Name: {model_name}\n"
                    missing_models_prompt += f"Project: {missing_model.get('project', 'Unknown')}\n"
                    
                    # Add columns if available
                    columns = missing_model.get("columns", [])
                    if columns:
                        missing_models_prompt += f"\nColumns ({len(columns)}):\n"
                        for column in columns[:20]:  # Limit to 20 columns to keep the prompt reasonable
                            col_name = column.get("name", "Unknown")
                            col_type = column.get("type", "Unknown")
                            missing_models_prompt += f"- {col_name} ({col_type})\n"
                        
                        if len(columns) > 20:
                            missing_models_prompt += f"... and {len(columns) - 20} more columns\n"
                    
                    # Add SQL if available
                    sql_code = missing_model.get("sql", "")
                    if sql_code:
                        sql_excerpt = sql_code[:1000] + "..." if len(sql_code) > 1000 else sql_code
                        missing_models_prompt += f"\nSQL Code snippet:\n{sql_excerpt}\n"
                
                # Add instructions for response format
                missing_models_prompt += """
                ---MODEL INFORMATION ENDS---
                
                TASK:
                For each model, provide:
                1. A comprehensive business-focused description of what this model represents.
                2. A confidence score from 1-5 for each description.
                
                For each model, format your response as follows:
                
                MODEL #1 DESCRIPTION:
                This is a comprehensive description of the model...
                
                MODEL #1 CONFIDENCE: 4
                
                MODEL #2 DESCRIPTION:
                ... description for model #2
                
                MODEL #2 CONFIDENCE: 3
                
                Continue this pattern for all models. Focus on business meaning and purpose.
                """
                
                # Make a single batch API request for all missing models
                print(f"Sending batch API request for {len(models_missing_descriptions)} missing models...")
                batch_response = self._make_api_request(missing_models_prompt, max_tokens=4000, temperature=0.1)
                
                if batch_response:
                    # Process each model in the response
                    for i, missing_model in enumerate(models_missing_descriptions):
                        model_name = missing_model.get("name", "Unknown")
                        
                        # Extract model description using regex
                        model_desc_pattern = rf"MODEL #{i+1} DESCRIPTION:\s*(.*?)(?:MODEL #{i+2} DESCRIPTION:|MODEL #{i+1} CONFIDENCE:|$)"
                        model_desc_match = re.search(model_desc_pattern, batch_response, re.DOTALL)
                        
                        if model_desc_match:
                            model_desc = model_desc_match.group(1).strip()
                            missing_model["ai_description"] = model_desc
                            if not missing_model.get("description"):
                                missing_model["description"] = model_desc
                                missing_model["user_edited"] = False
                                models_updated += 1
                                print(f"    - Added description for model {model_name}")
                        
                        # Extract model confidence score
                        model_conf_pattern = rf"MODEL #{i+1} CONFIDENCE:\s*(\d+)"
                        model_conf_match = re.search(model_conf_pattern, batch_response)
                        
                        if model_conf_match:
                            confidence = int(model_conf_match.group(1))
                            missing_model["ai_confidence_score"] = confidence
                            print(f"    - Added confidence score {confidence}/5 for model {model_name}")
                
                # Update the original models with the new descriptions
                for updated_model in models_missing_descriptions:
                    for i, model in enumerate(models):
                        if model.get("name") == updated_model.get("name"):
                            # Update only the model description, not the columns
                            models[i]["description"] = updated_model.get("description", "")
                            models[i]["ai_description"] = updated_model.get("ai_description", "")
                            models[i]["ai_confidence_score"] = updated_model.get("ai_confidence_score")
                            break
                            
            # Process models with missing column descriptions
            if models_with_missing_column_descriptions:
                # Count total columns without descriptions
                total_missing_columns = sum(len(data["missing_columns"]) for _, data in models_with_missing_column_descriptions.items())
                print(f"\n=== PROCESSING COLUMNS WITHOUT DESCRIPTIONS FOR {len(models_with_missing_column_descriptions)} MODELS ({total_missing_columns} COLUMNS) ===")
                
                # Create batches of models with missing columns, respecting the column limit per batch
                batches = []
                current_batch = []
                current_batch_columns = 0
                max_columns_per_batch = 50  # Smaller batch size for better handling of column descriptions
                
                for model_name, data in models_with_missing_column_descriptions.items():
                    model = data["model"]
                    missing_columns = data["missing_columns"]
                    
                    # Skip if no missing columns
                    if not missing_columns:
                        continue
                        
                    model_column_count = len(missing_columns)
                    
                    # If this model alone exceeds the column limit, process it individually
                    if model_column_count > max_columns_per_batch:
                        print(f"Model {model_name} has {model_column_count} columns without descriptions, processing individually.")
                        batches.append([{"model": model, "missing_columns": missing_columns}])
                        continue
                    
                    # If adding this model would exceed the column limit, start a new batch
                    if current_batch_columns + model_column_count > max_columns_per_batch:
                        batches.append(current_batch)
                        current_batch = [{"model": model, "missing_columns": missing_columns}]
                        current_batch_columns = model_column_count
                else:
                        # Add model to current batch
                        current_batch.append({"model": model, "missing_columns": missing_columns})
                        current_batch_columns += model_column_count
                
                # Add the last batch if it's not empty
                if current_batch:
                    batches.append(current_batch)
                
                print(f"Created {len(batches)} batches for processing missing column descriptions")
                
                # Process each batch
                for batch_index, batch in enumerate(batches):
                    batch_columns_count = sum(len(data["missing_columns"]) for data in batch)
                    print(f"\nProcessing batch {batch_index + 1}/{len(batches)} with {batch_columns_count} missing columns across {len(batch)} models")
                    
                    # Create a combined prompt for all models and columns in this batch
                    batch_prompt = "You are a data column documentation expert. I need descriptions for columns from multiple models:\n\n"
                    
                    # Add each model and its missing columns to the prompt
                    for model_index, data in enumerate(batch):
                        model = data["model"]
                        missing_columns = data["missing_columns"]
                        model_name = model.get("name", "Unknown")
                        
                        batch_prompt += f"---MODEL #{model_index+1}: {model_name}---\n"
                        batch_prompt += f"Project: {model.get('project', 'Unknown')}\n"
                        
                        # Add model description for context if available
                        if model.get("description") or model.get("ai_description"):
                            batch_prompt += f"Model Description: {model.get('description') or model.get('ai_description')}\n"
                        
                        # Add columns that need descriptions
                        batch_prompt += f"Columns ({len(missing_columns)}):\n"
                        for col_index, column in enumerate(missing_columns):
                            col_name = column.get("name", "Unknown")
                            col_type = column.get("type", "Unknown")
                            batch_prompt += f"- {model_index+1}.{col_index+1}: {col_name} ({col_type})\n"
                        
                        batch_prompt += "\n"
                    
                    # Add instructions for the response format
                    batch_prompt += """
                    TASK:
                    For each column in each model, provide a business-focused description that explains:
                    1. What data the column stores
                    2. Its purpose in the model
                    3. Its likely business significance
                    
                    Respond using this JSON format:
                    
                    {
                      "models": [
                        {
                          "model_index": 1,
                          "columns": [
                            {
                              "column_index": "1.1",
                              "name": "column_name",
                              "description": "Clear business description of what this column represents",
                              "confidence": 4
                            },
                            {
                              "column_index": "1.2",
                              "name": "another_column_name",
                              "description": "Clear business description of what this column represents",
                              "confidence": 3
                            }
                          ]
                        },
                        {
                          "model_index": 2,
                          "columns": [
                            ...
                          ]
                        }
                      ]
                    }
                    
                    For confidence scores, use:
                    5 = Very high confidence (Almost certain)
                    4 = High confidence (Strong certainty)
                    3 = Moderate confidence (Reasonable guess)
                    2 = Low confidence (Uncertain)
                    1 = Very low confidence (Highly uncertain)
                    
                    Make sure your response is valid JSON and includes ALL columns from ALL models.
                    """
                    
                    # Make API request for this batch of columns
                    print(f"Sending batch API request for {batch_columns_count} columns across {len(batch)} models...")
                    batch_response = self._make_api_request(batch_prompt, max_tokens=8000, temperature=0.1)
                    
                    if batch_response:
                        try:
                            # Extract and parse JSON
                            json_str = batch_response.strip()
                            json_start = json_str.find('{')
                            json_end = json_str.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = json_str[json_start:json_end]
                                
                            descriptions = json.loads(json_str)
                            
                            # Process the response
                            if "models" in descriptions and isinstance(descriptions["models"], list):
                                batch_columns_updated = 0
                                
                                # Process each model in the response
                                for model_data in descriptions["models"]:
                                    if "model_index" not in model_data or "columns" not in model_data:
                                        continue
                                        
                                    model_index = int(model_data["model_index"]) - 1  # Convert to 0-based
                                    if model_index < 0 or model_index >= len(batch):
                                        continue
                                        
                                    # Get the model info from our batch
                                    batch_model_data = batch[model_index]
                                    model = batch_model_data["model"]
                                    model_name = model.get("name", "Unknown")
                                    
                                    # Process each column
                                    for column_data in model_data.get("columns", []):
                                        if "name" not in column_data or "description" not in column_data:
                                            continue
                                            
                                        col_name = column_data["name"]
                                        col_desc = column_data["description"]
                                        col_conf = column_data.get("confidence", 3)
                                        
                                        # Find this column in the original model
                                        # First, make sure we're using the right models variable
                                        all_models = models  # Use the models passed from the batch_process_models method
                                        
                                        # Search in all models for the matching model and column
                                        for original_model in all_models:
                                            if original_model.get("name") == model_name:
                                                for column in original_model.get("columns", []):
                                                    if column.get("name") == col_name:
                                                        # Update the column description
                                                        column["ai_description"] = col_desc
                                                        
                                                        # If no description exists, use the AI one
                                                        if not column.get("description"):
                                                            column["description"] = col_desc
                                                            column["user_edited"] = False
                                                            batch_columns_updated += 1
                                                        
                                                        # Store confidence level
                                                        confidence = int(col_conf) if isinstance(col_conf, (int, str)) else 3
                                                        column["ai_confidence_score"] = confidence
                                                        column["needs_review"] = confidence <= 3
                                                        break
                                                break
                                
                                print(f"  - Updated {batch_columns_updated} columns in batch {batch_index + 1}")
                                columns_updated += batch_columns_updated
                            else:
                                print(f"  - Warning: Response format incorrect for batch {batch_index + 1}")
                                
                        except Exception as e:
                            print(f"  - Error processing batch {batch_index + 1}: {str(e)}")
                            if self.debug_mode:
                                import traceback
                                traceback.print_exc()
                    else:
                        print(f"  - Failed to get API response for batch {batch_index + 1}")
                        
                print(f"\nCompleted processing of all {len(batches)} batches for columns without descriptions")
                print(f"Updated a total of {columns_updated} columns with descriptions")
            
            # Calculate processing time
            processing_end_time = time.time()
            processing_time = processing_end_time - processing_start_time
            
            # Calculate coverage stats
            total_models = len(models_batch)
            models_with_desc = sum(1 for model in models_batch if model.get("description") or model.get("ai_description"))
            models_coverage = (models_with_desc / total_models) * 100 if total_models > 0 else 0
            
            columns_with_desc = sum(
                sum(1 for col in model.get("columns", []) if col.get("description") or col.get("ai_description"))
                for model in models_batch
            )
            columns_coverage = (columns_with_desc / total_columns) * 100 if total_columns > 0 else 0
            
            # Print batch summary
            print(f"\n=== Batch Processing Summary ===")
            print(f"Models updated: {models_with_desc}/{total_models} ({models_coverage:.1f}%)")
            print(f"Columns updated: {columns_with_desc}/{total_columns} ({columns_coverage:.1f}%)")
            cols_without_desc = total_columns - columns_with_desc
            print(f"Missing descriptions: {cols_without_desc}/{total_columns} columns still have no descriptions ({(cols_without_desc/total_columns*100) if total_columns > 0 else 0:.1f}%)")
            
            # Calculate and print model confidence scores
            print(f"\n=== Model Confidence Scores ===")
            for model in models_batch:
                model_name = model.get("name", "Unknown")
                confidence = model.get("ai_confidence_score")
                confidence_level = "No confidence score"
                if confidence is not None:
                    if confidence >= 4:
                        confidence_level = "High"
                    elif confidence >= 3:
                        confidence_level = "Good"
                    elif confidence >= 2:
                        confidence_level = "Moderate"
                    else:
                        confidence_level = "Low"
                        
                print(f"  - {model_name}: {confidence}/5 ({confidence_level})" if confidence is not None else f"  - {model_name}: {confidence_level}")
            
            # Calculate average confidence scores
            model_confidence_scores = [model.get("ai_confidence_score", 0) for model in models_batch if model.get("ai_confidence_score") is not None]
            column_confidence_scores = []
            for model in models_batch:
                for column in model.get("columns", []):
                    if column.get("ai_confidence_score") is not None:
                        column_confidence_scores.append(column.get("ai_confidence_score"))
            
            if model_confidence_scores:
                avg_model_conf = sum(model_confidence_scores) / len(model_confidence_scores)
                print(f"\nAverage batch model confidence score: {avg_model_conf:.2f}/5")
            
            if column_confidence_scores:
                avg_col_conf = sum(column_confidence_scores) / len(column_confidence_scores)
                print(f"Average column confidence score: {avg_col_conf:.2f}/5")
                
                # Column confidence distribution
                high_conf = sum(1 for score in column_confidence_scores if score >= 4)
                good_conf = sum(1 for score in column_confidence_scores if score == 3)
                moderate_conf = sum(1 for score in column_confidence_scores if score == 2)
                low_conf = sum(1 for score in column_confidence_scores if score <= 1)
                
                print(f"Column confidence distribution: {high_conf} High, {good_conf} Good, {moderate_conf} Moderate, {low_conf} Low")
            
            print(f"\nResponse processing completed in {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
            print(f"Total batch time (API + processing): {api_time + processing_time:.2f} seconds ({(api_time + processing_time)/60:.2f} minutes)")
            
            # Track models with missing descriptions for diagnostics
            try:
                models_with_missing_descriptions = []
                for model in models_batch:
                    model_name = model.get("name", "Unknown")
                    model_columns = model.get("columns", [])
                    missing_columns = []
                    
                    for column in model_columns:
                        if not column.get("description") and not column.get("ai_description"):
                            missing_columns.append({
                                "name": column.get("name", "Unknown"),
                                "type": column.get("type", "Unknown")
                            })
                    
                    if missing_columns:
                        model_entry = {
                            "id": model.get("id", "Unknown"),
                            "name": model_name,
                            "project": model.get("project", "Unknown"),
                            "missing_columns": missing_columns,
                            "total_columns": len(model_columns),
                            "columns_missing": len(missing_columns),
                            "missing_percentage": (len(missing_columns) / len(model_columns) * 100) if model_columns else 0,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Add sample API response for debugging if available
                        if model.get("_last_api_response"):
                            model_entry["response_excerpt"] = model.get("_last_api_response")[:2000]  # First 2000 chars
                        
                        models_with_missing_descriptions.append(model_entry)
            
                # Append models with missing descriptions to file immediately
                if models_with_missing_descriptions:
                    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "no_descriptions_found.json")
                    
                    # Load existing data if file exists
                    existing_data = {"models": []}
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r') as f:
                                existing_data = json.load(f)
                        except:
                            # If the file exists but can't be read, create a new one
                            existing_data = {"models": []}
                    
                    # Add new models to the existing list
                    existing_data["models"].extend(models_with_missing_descriptions)
                    existing_data["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Write back to file
                    with open(output_file, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                    
                    print(f"Appended {len(models_with_missing_descriptions)} models with missing descriptions to {output_file}")
            except Exception as e:
                print(f"Failed to append to missing descriptions file: {str(e)}")
                import traceback
                traceback.print_exc()
            
            return models_batch
                
        except Exception as e:
            print(f"Error processing batch descriptions: {str(e)}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return models_batch

    def enrich_metadata_efficiently(self, metadata: Dict[str, Any], manifest_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enrich metadata with AI-generated descriptions efficiently using advanced batching strategy"""
        import time
        import datetime
        start_time = time.time()
        
        # Skip if no models in metadata
        if "models" not in metadata or not metadata["models"]:
            print("No models in metadata, skipping AI description generation")
            return metadata
        
        print("\n=== STARTING EFFICIENT METADATA ENRICHMENT ===")
        print(f"Processing {len(metadata['models'])} models with efficient batch strategy")
        print("This approach significantly reduces API calls by processing whole models with their columns in a single call")
        print("Additionally, models are batched intelligently based on column count to maximize efficiency\n")
        
        # Process the models to add descriptions
        models = metadata.get("models", [])
        updated_models = self._process_models_in_batches(models, manifest_data)
        
        # Check for any models or columns without descriptions and process them
        print("\nChecking for any models or columns without descriptions...")
        self._process_missing_descriptions(updated_models, manifest_data)
        
        # Update the metadata with the enriched models
        metadata["models"] = updated_models
        
        # Track and report metrics
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n=== COMPLETED EFFICIENT METADATA ENRICHMENT ===")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return metadata

    def _process_models_in_batches(self, models: List[Dict[str, Any]], manifest_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process models in batches for efficient description generation.
        This method creates batches based on column count and processes them with the batch_process_models method.
        
        Args:
            models: List of models to process
            manifest_data: Optional manifest data for domain context
            
        Returns:
            List of processed models with descriptions
        """
        import time
        import datetime
        
        # Filter models that need processing
        models_to_process = []
        for model in models:
            model_name = model.get("name", "Unknown")
            model_column_count = len(model.get("columns", []))
            
            # Skip models that already have descriptions unless forced
            if model.get("description") and not model.get("refresh_description"):
                print(f"Skipping model: {model_name} - already has description")
                continue
                
            # Include all models for processing, even those with no columns
            models_to_process.append(model)
            
        # Create batches based on column count
        max_columns_per_request = 200  # Maximum columns per API request with 16000 token limit
        
        # Create batches by column count rather than fixed model count
        batches = []
        current_batch = []
        current_batch_columns = 0
        
        for model in models_to_process:
            # Consider models with no columns as having at least 1 column for batching purposes
            model_column_count = max(1, len(model.get("columns", [])))
            
            # If this model alone exceeds the column limit, process it individually
            if model_column_count > max_columns_per_request:
                print(f"Model {model.get('name', 'Unknown')} has {model_column_count} columns, exceeding the limit.")
                print(f"It will be processed individually to maintain context integrity.")
                batches.append([model])
                continue
                
            # If adding this model would exceed the column limit, start a new batch
            if current_batch_columns + model_column_count > max_columns_per_request:
                batches.append(current_batch)
                current_batch = [model]
                current_batch_columns = model_column_count
            else:
                # Add model to current batch
                current_batch.append(model)
                current_batch_columns += model_column_count
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
            
        print(f"Created {len(batches)} batches based on column count limit of {max_columns_per_request}:")
        for i, batch in enumerate(batches):
            # Calculate batch column count, treating models with no columns as having 1 column
            batch_column_count = sum(max(1, len(model.get("columns", []))) for model in batch)
            model_names = [model.get("name", "Unknown") for model in batch]
            print(f"  Batch {i+1}: {len(batch)} models, {batch_column_count} columns - {', '.join(model_names)}")
        
        # Track metrics
        models_processed = 0
        
        for i, batch in enumerate(batches):
            print(f"\nProcessing batch {i+1}/{len(batches)}")
            # Calculate batch column count, treating models with no columns as having 1 column
            batch_column_count = sum(max(1, len(model.get("columns", []))) for model in batch)
            print(f"Batch contains {len(batch)} models with {batch_column_count} columns total")
            
            # Start timing this batch
            batch_start_time = time.time()
            
            # Process this batch of models
            updated_batch = self.batch_process_models(batch, manifest_data)
            
            # Calculate and display time taken for this batch
            batch_end_time = time.time()
            batch_time_taken = batch_end_time - batch_start_time
            print(f"Batch {i+1} processing completed in {batch_time_taken:.2f} seconds ({batch_time_taken/60:.2f} minutes)")
            
            # Update the models in the original list
            for updated_model in updated_batch:
                model_name = updated_model.get("name", "Unknown")
                # Find and update the corresponding model in the full list
                for j, model in enumerate(models):
                    if model.get("name") == model_name:
                        models[j] = updated_model
                        break
            
            models_processed += len(batch)
            print(f"Progress: Processed {models_processed}/{len(models_to_process)} models")
            
            # Calculate and print confidence scores for this batch
            batch_model_confidence_scores = []
            batch_column_confidence_scores = []
            
            # Collect confidence scores for this batch
            for model in updated_batch:
                if model.get("ai_confidence_score") is not None:
                    batch_model_confidence_scores.append(model.get("ai_confidence_score"))
                
                for column in model.get("columns", []):
                    if column.get("ai_confidence_score") is not None:
                        batch_column_confidence_scores.append(column.get("ai_confidence_score"))
            
            # Print average batch confidence scores
            if batch_model_confidence_scores:
                avg_model_conf = sum(batch_model_confidence_scores) / len(batch_model_confidence_scores)
                print(f"Average model confidence score: {avg_model_conf:.2f}/5")
            
            if batch_column_confidence_scores:
                avg_col_conf = sum(batch_column_confidence_scores) / len(batch_column_confidence_scores)
                print(f"Average column confidence score: {avg_col_conf:.2f}/5")
        
        return models

    def _process_missing_descriptions(self, models: List[Dict[str, Any]], manifest_data: Dict[str, Any] = None) -> None:
        """
        Process models and columns that are still missing descriptions after batch processing.
        This function identifies models and columns without descriptions and processes them
        in a special API request to ensure complete coverage.
        
        Args:
            models: List of model dictionaries that have been processed
            manifest_data: Optional manifest data for domain context
        """
        import time
        
        # Track metrics for detailed reporting
        start_time = time.time()
        total_api_time = 0
        total_processing_time = 0
        models_updated = 0
        columns_updated = 0
        all_model_confidence_scores = []
        all_column_confidence_scores = []
        
        # Identify models and columns without descriptions
        models_without_descriptions = []
        models_with_columns_without_descriptions = []
        
        for model in models:
            model_name = model.get("name", "Unknown")
            model_has_description = bool(model.get("description") or model.get("ai_description"))
            
            # Check columns without descriptions
            columns_without_descriptions = []
            for column in model.get("columns", []):
                if not (column.get("description") or column.get("ai_description")):
                    columns_without_descriptions.append(column)
            
            # Track models based on missing descriptions
            if not model_has_description and not columns_without_descriptions:
                models_without_descriptions.append(model)
            elif columns_without_descriptions:
                # Clone the model but only include columns without descriptions
                model_with_missing_columns = {
                    "id": model.get("id"),
                    "name": model.get("name"),
                    "project": model.get("project"),
                    "description": model.get("description", ""),
                    "ai_description": model.get("ai_description", ""),
                    "columns": columns_without_descriptions,
                    "sql": model.get("sql", "")
                }
                models_with_columns_without_descriptions.append(model_with_missing_columns)
        
        print(f"\n=== MISSING DESCRIPTIONS ANALYSIS ===")
        print(f"Found {len(models_without_descriptions)} models without any description")
        
        total_missing_columns = sum(len(model.get("columns", [])) for model in models_with_columns_without_descriptions)
        print(f"Found {total_missing_columns} columns without descriptions across {len(models_with_columns_without_descriptions)} models")
        
        # Process models without descriptions
        if models_without_descriptions:
            print(f"\n=== PROCESSING {len(models_without_descriptions)} MODELS WITHOUT DESCRIPTIONS ===")
            
            # Track processing time
            models_api_start = time.time()
            updated_models = self.batch_process_models(models_without_descriptions, manifest_data, all_models=models)
            models_api_end = time.time()
            api_time = models_api_end - models_api_start
            total_api_time += api_time
            
            print(f"API request for {len(models_without_descriptions)} models completed in {api_time:.2f} seconds ({api_time/60:.2f} minutes)")
            
            # Start processing time
            processing_start = time.time()
            
            # Update the original models with the new descriptions
            for updated_model in updated_models:
                for i, model in enumerate(models):
                    if model.get("name") == updated_model.get("name"):
                        # Update only the model description, not the columns
                        models[i]["description"] = updated_model.get("description", "")
                        models[i]["ai_description"] = updated_model.get("ai_description", "")
                        models[i]["ai_confidence_score"] = updated_model.get("ai_confidence_score")
                        models[i]["user_edited"] = updated_model.get("user_edited", False)
                        
                        # Track confidence score for reporting
                        if updated_model.get("ai_confidence_score") is not None:
                            all_model_confidence_scores.append(updated_model.get("ai_confidence_score"))
                            models_updated += 1
                        break
            
            processing_end = time.time()
            processing_time = processing_end - processing_start
            total_processing_time += processing_time
            
            # Report metrics for models batch
            print(f"\n=== MODEL DESCRIPTIONS PROCESSING SUMMARY ===")
            print(f"Models updated: {models_updated}/{len(models_without_descriptions)} ({(models_updated/len(models_without_descriptions)*100):.1f}%)")
            
            # Report confidence scores
            if all_model_confidence_scores:
                avg_confidence = sum(all_model_confidence_scores) / len(all_model_confidence_scores)
                print(f"Average model confidence score: {avg_confidence:.2f}/5")
                
                # Confidence distribution
                high_conf = sum(1 for score in all_model_confidence_scores if score >= 4)
                good_conf = sum(1 for score in all_model_confidence_scores if score == 3)
                moderate_conf = sum(1 for score in all_model_confidence_scores if score == 2)
                low_conf = sum(1 for score in all_model_confidence_scores if score <= 1)
                
                print(f"Model confidence distribution: {high_conf} High, {good_conf} Good, {moderate_conf} Moderate, {low_conf} Low")
                
                # Report percentage that need review
                need_review = sum(1 for score in all_model_confidence_scores if score <= 3)
                if need_review > 0:
                    print(f"{need_review} out of {len(all_model_confidence_scores)} models ({(need_review/len(all_model_confidence_scores)*100):.1f}%) may need human review (confidence <= 3)")
            
            print(f"Processing completed in {processing_time:.2f} seconds")
            print(f"Total time for models without descriptions: {api_time + processing_time:.2f} seconds")
        
        # Process models with columns without descriptions in batches of up to 200 columns
        if models_with_columns_without_descriptions:
            # Count total columns without descriptions
            total_missing_columns = sum(len(model.get("columns", [])) for model in models_with_columns_without_descriptions)
            print(f"\n=== PROCESSING {len(models_with_columns_without_descriptions)} MODELS WITH {total_missing_columns} MISSING COLUMN DESCRIPTIONS ===")
            
            # Create batches of models with missing columns, respecting the 200 column limit
            batches = []
            current_batch = []
            current_batch_columns = 0
            max_columns_per_batch = 200
            
            for model in models_with_columns_without_descriptions:
                model_column_count = len(model.get("columns", []))
                
                # If this model alone exceeds the column limit, process it individually
                if model_column_count > max_columns_per_batch:
                    print(f"Model {model.get('name')} has {model_column_count} columns without descriptions, processing individually.")
                    batches.append([model])
                    continue
                
                # If adding this model would exceed the column limit, start a new batch
                if current_batch_columns + model_column_count > max_columns_per_batch:
                    batches.append(current_batch)
                    current_batch = [model]
                    current_batch_columns = model_column_count
                else:
                    # Add model to current batch
                    current_batch.append(model)
                    current_batch_columns += model_column_count
            
            # Add the last batch if it's not empty
            if current_batch:
                batches.append(current_batch)
            
            # Process each batch
            batch_columns_updated = 0
            batch_confidence_scores = []
            
            for batch_index, batch in enumerate(batches):
                print(f"Processing batch {batch_index + 1}/{len(batches)} of models with missing column descriptions")
                batch_column_count = sum(len(model.get("columns", [])) for model in batch)
                print(f"Batch contains {len(batch)} models with {batch_column_count} columns missing descriptions")
                
                # Track API request time
                batch_api_start = time.time()
                
                # Process this batch
                try:
                    updated_batch = self.batch_process_models(batch, manifest_data, all_models=models)
                    
                    batch_api_end = time.time()
                    batch_api_time = batch_api_end - batch_api_start
                    total_api_time += batch_api_time
                    
                    print(f"API request completed in {batch_api_time:.2f} seconds ({batch_api_time/60:.2f} minutes)")
                    
                    # Track processing time
                    batch_processing_start = time.time()
                    
                    # Counters for this batch
                    columns_updated_in_batch = 0
                    batch_column_confidence_scores = []
                    
                    # Update the original models with the new column descriptions
                    for updated_model in updated_batch:
                        updated_model_name = updated_model.get("name")
                        
                        # Find the original model
                        for i, model in enumerate(models):
                            if model.get("name") == updated_model_name:
                                # For each column in the updated model, update the corresponding column in the original model
                                for updated_column in updated_model.get("columns", []):
                                    updated_column_name = updated_column.get("name")
                                    
                                    # Find and update the column in the original model
                                    for j, original_column in enumerate(models[i].get("columns", [])):
                                        if original_column.get("name") == updated_column_name:
                                            # Only update if the column didn't have a description before
                                            if not (original_column.get("description") or original_column.get("ai_description")):
                                                models[i]["columns"][j]["description"] = updated_column.get("description", "")
                                                models[i]["columns"][j]["ai_description"] = updated_column.get("ai_description", "")
                                                models[i]["columns"][j]["ai_confidence_score"] = updated_column.get("ai_confidence_score")
                                                models[i]["columns"][j]["user_edited"] = updated_column.get("user_edited", False)
                                                models[i]["columns"][j]["needs_review"] = updated_column.get("needs_review", False)
                                                
                                                # Track updates and confidence scores
                                                columns_updated_in_batch += 1
                                                if updated_column.get("ai_confidence_score") is not None:
                                                    confidence = updated_column.get("ai_confidence_score")
                                                    batch_column_confidence_scores.append(confidence)
                                                    all_column_confidence_scores.append(confidence)
                                            break
                                break
                    
                    batch_processing_end = time.time()
                    batch_processing_time = batch_processing_end - batch_processing_start
                    total_processing_time += batch_processing_time
                    
                    # Update batch metrics
                    batch_columns_updated += columns_updated_in_batch
                    columns_updated += columns_updated_in_batch
                    batch_confidence_scores.extend(batch_column_confidence_scores)
                    
                    # Report on this batch
                    print(f"Updated {columns_updated_in_batch}/{batch_column_count} columns in batch {batch_index + 1}")
                    
                    if batch_column_confidence_scores:
                        avg_batch_conf = sum(batch_column_confidence_scores) / len(batch_column_confidence_scores)
                        print(f"Average column confidence score for this batch: {avg_batch_conf:.2f}/5")
                    
                    print(f"Batch processing completed in {batch_processing_time:.2f} seconds")
                    print(f"Total batch time: {batch_api_time + batch_processing_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error processing batch descriptions: {str(e)}")
                    if self.debug_mode:
                        import traceback
                        traceback.print_exc()
            
                    # Print progress
                    print(f"Completed batch {batch_index + 1}/{len(batches)} of models with missing column descriptions")
            
            # Report metrics for columns processing
            if batch_columns_updated > 0:
                print(f"\n=== COLUMN DESCRIPTIONS PROCESSING SUMMARY ===")
                print(f"Total columns updated: {batch_columns_updated}/{total_missing_columns} ({(batch_columns_updated/total_missing_columns*100):.1f}%)")
                
                if batch_confidence_scores:
                    avg_confidence = sum(batch_confidence_scores) / len(batch_confidence_scores)
                    print(f"Average column confidence score: {avg_confidence:.2f}/5")
                    
                    # Confidence distribution
                    high_conf = sum(1 for score in batch_confidence_scores if score >= 4)
                    good_conf = sum(1 for score in batch_confidence_scores if score == 3)
                    moderate_conf = sum(1 for score in batch_confidence_scores if score == 2)
                    low_conf = sum(1 for score in batch_confidence_scores if score <= 1)
                    
                    print(f"Column confidence distribution: {high_conf} High, {good_conf} Good, {moderate_conf} Moderate, {low_conf} Low")
                    
                    # Report percentage that need review
                    need_review = sum(1 for score in batch_confidence_scores if score <= 3)
                    if need_review > 0:
                        print(f"{need_review} out of {len(batch_confidence_scores)} columns ({(need_review/len(batch_confidence_scores)*100):.1f}%) may need human review (confidence <= 3)")
        
        # Complete overall summary
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print final summary
        total_models = len(models)
        models_with_descriptions = sum(1 for model in models if model.get("description") or model.get("ai_description"))
        
        total_columns = sum(len(model.get("columns", [])) for model in models)
        columns_with_descriptions = sum(
            sum(1 for col in model.get("columns", []) if col.get("description") or col.get("ai_description"))
            for model in models
        )
        
        print("\n=== MISSING DESCRIPTIONS PROCESSING SUMMARY ===")
        print(f"Models with descriptions: {models_with_descriptions}/{total_models} ({models_with_descriptions/total_models*100:.1f}%)")
        print(f"Columns with descriptions: {columns_with_descriptions}/{total_columns} ({columns_with_descriptions/total_columns*100:.1f}%)")
        
        # Overall confidence scores
        if all_model_confidence_scores or all_column_confidence_scores:
            print("\n=== OVERALL CONFIDENCE SCORES ===")
            
            if all_model_confidence_scores:
                avg_model_conf = sum(all_model_confidence_scores) / len(all_model_confidence_scores)
                print(f"Average model confidence score: {avg_model_conf:.2f}/5")
            
            if all_column_confidence_scores:
                avg_col_conf = sum(all_column_confidence_scores) / len(all_column_confidence_scores)
                print(f"Average column confidence score: {avg_col_conf:.2f}/5")
        
        # Processing metrics
        print(f"\nTotal API request time: {total_api_time:.2f} seconds ({total_api_time/60:.2f} minutes)")
        print(f"Total processing time: {total_processing_time:.2f} seconds ({total_processing_time/60:.2f} minutes)")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Missing descriptions warning
        if total_models > models_with_descriptions or total_columns > columns_with_descriptions:
            print("\nWARNING: Some models or columns still don't have descriptions after processing")
            print("This may be due to API errors or issues with the response format.")
            print("Check the logs and exported metadata for details.")

    def enrich_metadata(self, metadata: Dict[str, Any], manifest_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enrich metadata with AI-generated descriptions using manifest data for domain context.
        This is a wrapper around enrich_metadata_efficiently for backward compatibility.
        """
        print("Using the efficient batch processing method for metadata enrichment")
        # Call our single-phase efficient implementation
        return self.enrich_metadata_efficiently(metadata, manifest_data)
        
   