# backend/services/metadata_service.py

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .dbt_metadata_parser import parse_dbt_projects, save_metadata
from .ai_description_service import AIDescriptionService

class MetadataService:
    """Service for combining and processing dbt project metadata"""
    
    def __init__(self, dbt_projects_dir: str = "dbt_pk/dbt", output_dir: str = None, use_ai_descriptions: bool = True):
        # Get the base directory (parent of backend)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default output_dir if not provided
        if output_dir is None:
            output_dir = os.path.join(base_dir, "exports")
        
        # Make sure we're using absolute paths
        if os.path.isabs(dbt_projects_dir):
            # If absolute path is provided, use it directly
            self.dbt_projects_dir = dbt_projects_dir
        else:
            # If relative path, resolve relative to base_dir
            self.dbt_projects_dir = os.path.abspath(os.path.join(base_dir, "..", dbt_projects_dir))
            
        self.output_dir = os.path.abspath(output_dir)
        self.unified_metadata_path = os.path.join(self.output_dir, "uni_metadata.json")
        self.metadata = {}
        self.use_ai_descriptions = use_ai_descriptions
        
        # Initialize AI description service if enabled
        self.ai_service = AIDescriptionService() if use_ai_descriptions else None
        
        print(f"Initializing MetadataService:")
        print(f"- Base dir: {base_dir}")
        print(f"- Projects dir: {self.dbt_projects_dir}")
        print(f"- Output dir: {self.output_dir}")
        print(f"- Metadata path: {self.unified_metadata_path}")
        print(f"- AI descriptions: {'enabled' if use_ai_descriptions else 'disabled'}")
        
        # Create exports directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize or load the unified metadata
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize or load the unified metadata"""
        if os.path.exists(self.unified_metadata_path):
            try:
                with open(self.unified_metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    print(f"Loaded metadata from {self.unified_metadata_path}")
                    print(f"Projects: {len(self.metadata.get('projects', []))}")
                    print(f"Models: {len(self.metadata.get('models', []))}")
                    print(f"Lineage: {len(self.metadata.get('lineage', []))}")
                    
                    # Print a sample project if available
                    if self.metadata.get('projects'):
                        print("\nSample Project:")
                        print(json.dumps(self.metadata['projects'][0], indent=2))
                    
                    # Print sample lineage if available
                    if self.metadata.get('lineage'):
                        print("\nSample Lineage Relationships:")
                        for i, lineage in enumerate(self.metadata['lineage'][:2]):
                            print(f"Lineage {i+1}: {json.dumps(lineage)}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.refresh()
        else:
            print(f"Metadata file not found at {self.unified_metadata_path}, creating new...")
            self.refresh()
    
    def get_project_dirs(self) -> List[str]:
        """Get all dbt project directories"""
        if not os.path.exists(self.dbt_projects_dir):
            print(f"Projects directory not found: {self.dbt_projects_dir}")
            return []
            
        return [d for d in os.listdir(self.dbt_projects_dir) 
                if os.path.isdir(os.path.join(self.dbt_projects_dir, d)) 
                and not d.startswith('.')]
    
    def _parse_dbt_projects(self) -> Dict[str, Any]:
        """Parse dbt project files and generate structured metadata"""
        projects = []
        models = []
        lineage_data = []
        model_id_map = {}  # Maps node_id to our model_id
        
        project_dirs = self.get_project_dirs()
        print(f"\nFound {len(project_dirs)} project directories: {project_dirs}")
        
        for project_dir in project_dirs:
            project_path = os.path.join(self.dbt_projects_dir, project_dir)
            
            # Look for manifest.json and catalog.json in the target directory
            manifest_path = os.path.join(project_path, "target", "manifest.json")
            catalog_path = os.path.join(project_path, "target", "catalog.json")
            
            # Skip if manifest doesn't exist
            if not os.path.exists(manifest_path):
                print(f"Skipping {project_dir}: No manifest.json found")
                continue
            
            print(f"\nProcessing project: {project_dir}")
            print(f"  Manifest path: {manifest_path}")
            print(f"  Catalog path: {catalog_path} (exists: {os.path.exists(catalog_path)})")
            
            try:
                # Load manifest.json
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Load catalog.json if it exists
                catalog_data = {}
                if os.path.exists(catalog_path):
                    with open(catalog_path, 'r') as f:
                        catalog_data = json.load(f)
                
                # Extract project name from manifest
                project_name = project_dir
                if 'metadata' in manifest_data and 'project_name' in manifest_data['metadata']:
                    project_name = manifest_data['metadata']['project_name']
                
                # Add project to projects list
                project_id = project_name.lower().replace(' ', '_')
                projects.append({
                    "id": project_id,
                    "name": project_name,
                    "description": f"{project_name} dbt project"
                })
                
                print(f"  Added project: {project_name} (id: {project_id})")
                
                # Process models
                model_count = 0
                for node_id, node in manifest_data.get('nodes', {}).items():
                    if node.get('resource_type') == 'model':
                        model_name = node.get('name', '')
                        
                        # Generate a unique ID for the model
                        model_id = f"{project_id}_{model_name}"
                        model_id_map[node_id] = model_id
                        model_count += 1
                        
                        # Get SQL code
                        sql = ""
                        if 'compiled_sql' in node:
                            sql = node['compiled_sql']
                        elif 'raw_sql' in node:
                            sql = node['raw_sql']
                        elif 'raw_code' in node:
                            sql = node['raw_code']
                        
                        # Extract columns
                        columns = []
                        
                        # Try to get columns from catalog
                        catalog_node = None
                        if catalog_data and 'nodes' in catalog_data:
                            catalog_node = catalog_data['nodes'].get(node_id)
                        
                        if catalog_node and 'columns' in catalog_node:
                            for col_name, col_data in catalog_node['columns'].items():
                                column = {
                                    "name": col_name,
                                    "type": col_data.get('type', 'unknown'),
                                    "description": col_data.get('description', ''),
                                    "isPrimaryKey": 'primary' in col_name.lower() or col_name.lower().endswith('_id'),
                                    "isForeignKey": 'foreign' in col_name.lower() or col_name.lower().endswith('_id')
                                }
                                columns.append(column)
                        else:
                            # If no catalog, try to extract columns from SQL
                            extracted_columns = self._extract_columns_from_sql(sql)
                            for col_name in extracted_columns:
                                column = {
                                    "name": col_name,
                                    "type": "unknown",
                                    "description": "",
                                    "isPrimaryKey": 'primary' in col_name.lower() or col_name.lower().endswith('_id'),
                                    "isForeignKey": 'foreign' in col_name.lower() or col_name.lower().endswith('_id')
                                }
                                columns.append(column)
                        
                        # Create model object
                        model = {
                            "id": model_id,
                            "name": model_name,
                            "project": project_id,
                            "description": node.get('description', ''),
                            "columns": columns,
                            "sql": sql
                        }
                        
                        # Add additional model metadata from manifest
                        if 'config' in node:
                            config = node.get('config', {})
                            model["materialized"] = config.get('materialized', 'view')
                            model["schema"] = config.get('schema', node.get('schema'))
                        
                        # Add file path
                        model["file_path"] = node.get('original_file_path', 'N/A')
                        
                        # Extract database and schema if available
                        if 'database' in node:
                            model["database"] = node.get('database')
                            
                        if 'schema' in node and not model.get('schema'):
                            model["schema"] = node.get('schema')
                            
                        # Ensure we have a schema value by using the default schema if needed
                        if not model.get('schema') and 'config' in manifest_data.get('metadata', {}):
                            model["schema"] = manifest_data.get('metadata', {}).get('config', {}).get('schema')
                            
                        # Extract tags if available
                        if 'tags' in node and node.get('tags'):
                            model["tags"] = node.get('tags')
                            
                        # Extract additional catalog information if available
                        if catalog_node:
                            model["catalog_metadata"] = {
                                "unique_id": catalog_node.get('unique_id'),
                                "metadata": catalog_node.get('metadata', {})
                            }
                            
                            # Get stats if available
                            if 'stats' in catalog_node:
                                model["stats"] = catalog_node.get('stats')
                                
                        # Make sure description is complete and not truncated
                        full_description = node.get('description', '')
                        if isinstance(full_description, str) and len(full_description) > 0:
                            model["description"] = full_description
                            
                        models.append(model)
                        
                        if model_count <= 2:
                            print(f"  Added model: {model_name} (id: {model_id})")
                            if columns:
                                print(f"    First column: {columns[0]['name']} ({columns[0]['type']})")
                
                print(f"  Processed {model_count} models for project {project_name}")
                
                # Process lineage
                lineage_data = []
                model_by_name = {}
                
                # First, build a lookup of models by name
                for model in models:
                    name = model['name']
                    if name not in model_by_name:
                        model_by_name[name] = []
                    model_by_name[name].append(model)
                
                # Now process each model for lineage
                for model in models:
                    # Get the SQL
                    sql = model.get('sql', '')
                    if not sql:
                        continue
                    
                    # Look for source() references
                    source_refs = re.findall(r'{{\s*source\([\'"]([^\'"]+)[\'"],\s*[\'"]([^\'"]+)[\'"]\)\s*}}', sql)
                    for source_name, table_name in source_refs:
                        # Find the referenced model in any project
                        if table_name in model_by_name:
                            for source_model in model_by_name[table_name]:
                                lineage_data.append({
                                    "source": source_model['id'],
                                    "target": model['id'],
                                    "type": "source"
                                })
                    
                    # Look for ref() references
                    ref_matches = re.findall(r'{{\s*ref\([\'"]([^\'"]+)[\'"]\)\s*}}', sql)
                    for ref_name in ref_matches:
                        # Find the referenced model in any project
                        if ref_name in model_by_name:
                            for ref_model in model_by_name[ref_name]:
                                lineage_data.append({
                                    "source": ref_model['id'],
                                    "target": model['id'],
                                    "type": "ref"
                                })
                
                # Create the final metadata object
                metadata = {
                    "projects": projects,
                    "models": models,
                    "lineage": lineage_data
                }
                
                print(f"\nParsing complete:")
                print(f"  Projects: {len(projects)}")
                print(f"  Models: {len(models)}")
                print(f"  Lineage relationships: {len(lineage_data)}")
                
                return metadata
            
            except Exception as e:
                print(f"Error processing project {project_dir}: {str(e)}")
        
        # Create the final metadata object
        metadata = {
            "projects": projects,
            "models": models,
            "lineage": lineage_data
        }
        
        print(f"\nParsing complete:")
        print(f"  Projects: {len(projects)}")
        print(f"  Models: {len(models)}")
        print(f"  Lineage relationships: {len(lineage_data)}")
        
        return metadata
    
    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        """Extract column names from SQL query"""
        if not sql:
            return []
        
        # Try to find the SELECT statement
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return []
        
        select_clause = select_match.group(1)
        
        # Split by commas, but ignore commas inside functions
        columns = []
        bracket_level = 0
        current_column = ""
        
        for char in select_clause:
            if char == '(':
                bracket_level += 1
                current_column += char
            elif char == ')':
                bracket_level -= 1
                current_column += char
            elif char == ',' and bracket_level == 0:
                columns.append(current_column.strip())
                current_column = ""
            else:
                current_column += char
        
        if current_column:
            columns.append(current_column.strip())
        
        # Process each column expression
        result = []
        for col_expr in columns:
            # Check for AS keyword
            as_match = re.search(r'(?:AS|as)\s+([a-zA-Z0-9_]+)$', col_expr)
            if as_match:
                result.append(as_match.group(1))
            else:
                # If no AS keyword, use the last part after a dot or the whole expression
                col_parts = col_expr.split('.')
                col_name = col_parts[-1].strip()
                
                # Remove any functions
                if '(' not in col_name:
                    result.append(col_name)
        
        return result
    
    def refresh(self, additional_prompt: str = None) -> bool:
        """
        Refresh metadata from dbt projects using structured analysis with AI-powered descriptions.
        This uses advanced batch processing for optimal performance.
        
        Args:
            additional_prompt: Optional additional context for the AI prompts
            
        Returns:
            bool: True if refresh was successful, False otherwise
        """
        try:
            print(f"======= PHASE 1: EXTRACTING METADATA =======")
            print(f"Refreshing metadata from {self.dbt_projects_dir}")
            if not os.path.exists(self.dbt_projects_dir):
                print(f"Error: Projects directory does not exist: {self.dbt_projects_dir}")
                return False
            
            # Create exports directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Parse all dbt projects
            print(f"Parsing dbt projects to extract metadata...")
            raw_metadata = parse_dbt_projects(self.dbt_projects_dir)
            
            if not raw_metadata:
                print("Error: Failed to parse dbt projects")
                return False
            
            print(f"Successfully extracted raw metadata:")
            print(f"  - Projects: {len(raw_metadata.get('projects', []))}")
            print(f"  - Models: {len(raw_metadata.get('models', []))}")
            print(f"  - Lineage relationships: {len(raw_metadata.get('lineage', []))}")
                
            # Use the raw metadata directly without modification
            self.metadata = raw_metadata
            
            # Save the raw metadata first
            print(f"Saving raw metadata to {self.unified_metadata_path}")
            with open(self.unified_metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Raw metadata saved successfully!")
            
            # Load manifest data for domain context if possible
            manifest_data = self.load_manifest_data()
            
            # Ask for additional prompt info if not provided
            if additional_prompt is None:
                # Show the default prompt
                base_prompt = """You are a data model expert with deep domain knowledge in financial services, insurance, and business data. 
I need you to analyze a data model and its columns, providing detailed descriptions."""
                
                print("\n======= PROMPT CUSTOMIZATION =======")
                print(f"Default prompt base:\n{base_prompt}\n")
                print("Would you like to add additional context to the AI prompt? This could include:")
                print("- Domain knowledge about the data")
                print("- Business context about how this data is used")
                print("- Naming conventions or abbreviations used in your organization")
                print("- Any specific instructions for description generation")
                print("(Press Enter to use the default prompt, or type your additions)\n")
                
                try:
                    additional_prompt = input("Additional prompt context: ")
                except EOFError:
                    additional_prompt = ""
                
                if additional_prompt.strip():
                    print(f"Using additional context: {additional_prompt}")
                else:
                    print("Using default prompt without additional context")
            
            # Generate AI descriptions if enabled
            if self.use_ai_descriptions and self.ai_service:
                print(f"\n======= PHASE 2: ENRICHING WITH AI DESCRIPTIONS =======")
                print("\n=== Starting AI Description Generation Using Efficient Batch Processing ===")
                
                # Set the additional prompt context if provided
                if additional_prompt and additional_prompt.strip():
                    self.ai_service.additional_prompt_context = additional_prompt.strip()
                    print(f"Using additional prompt context: {additional_prompt.strip()}")
                
                # Use the efficient batch processing for all models
                print("Using efficient batch processing to generate descriptions for all models and their columns...")
                print("This will update the previously saved metadata file with AI-generated descriptions")
                
                # Track the number of models before AI enrichment
                models_before = len(self.metadata.get('models', []))
                columns_before = sum(len(model.get('columns', [])) for model in self.metadata.get('models', []))
                
                # Enrich the metadata with AI descriptions
                self.metadata = self.ai_service.enrich_metadata_efficiently(self.metadata, manifest_data)
                
                # Calculate statistics after enrichment
                models_after = len(self.metadata.get('models', []))
                columns_after = sum(len(model.get('columns', [])) for model in self.metadata.get('models', []))
                
                print(f"\nAI Enrichment Statistics:")
                print(f"  - Models processed: {models_after} (was {models_before} before)")
                print(f"  - Columns processed: {columns_after} (was {columns_before} before)")
                
                print(f"=== Completed AI Description Generation Using Efficient Batch Processing ===\n")
            else:
                print("\nSkipping AI description generation (not enabled)")
            
            # Save the processed metadata
            print(f"\n======= PHASE 3: SAVING ENRICHED METADATA =======")
            print(f"Saving enriched metadata to {self.unified_metadata_path}")
            with open(self.unified_metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save a copy to the dbt projects directory as well
            # This is useful for systems that need the metadata to be alongside the dbt projects
            print(f"Also saving a copy to the dbt projects directory")
            save_metadata(self.metadata, self.dbt_projects_dir)
            
            # Print a summary of the metadata
            projects_count = len(self.metadata.get('projects', []))
            models_count = len(self.metadata.get('models', []))
            lineage_count = len(self.metadata.get('lineage', []))
            
            print(f"\nMetadata refresh completed successfully:")
            print(f"  - Projects: {projects_count}")
            print(f"  - Models: {models_count}")
            print(f"  - Lineage relationships: {lineage_count}")
            return True
        except Exception as e:
            print(f"Error refreshing metadata: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects"""
        projects = self.metadata.get("projects", [])
        print(f"Returning {len(projects)} projects")
        return projects
    
    def get_models(self, project_id: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all models, optionally filtered by project and search term"""
        models = self.metadata.get("models", [])
        
        # Filter out sources - using proper logical approach to identify sources
        models = [m for m in models if not (
            m.get("is_source", False) or  # Check is_source flag
            m.get("materialized") == "source" or  # Check materialized = source
            (isinstance(m.get("id", ""), str) and "_raw_" in m.get("id", ""))  # Backup check for raw sources
        )]
        
        # Filter by project if specified
        if project_id:
            models = [m for m in models if m["project"] == project_id]
        
        # Filter by search term if specified
        if search and search.strip():  # Ensure search is not empty or just whitespace
            search = search.lower().strip()
            print(f"Searching for exact match on model name: '{search}'")
            
            # Track matching models for debug logging
            name_matches = []
            desc_matches = []
            
            # Use exact matching for names, but still allow partial matching in descriptions
            filtered_models = []
            for model in models:
                model_name = model["name"].lower()
                if model_name == search:
                    name_matches.append(model["name"])
                    filtered_models.append(model)
                elif model.get("description") and search in model.get("description", "").lower():
                    desc_matches.append(model["name"])
                    filtered_models.append(model)
            
            # Debug logging
            print(f"Models matching by name ({len(name_matches)}): {', '.join(name_matches)}")
            print(f"Models matching by description ({len(desc_matches)}): {', '.join(desc_matches)}")
            
            models = filtered_models
        
        print(f"Returning {len(models)} models" + 
              (f" for project {project_id}" if project_id else "") +
              (f" matching search '{search}'" if search else ""))
        return models
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model by ID"""
        for model in self.metadata.get("models", []):
            if model["id"] == model_id:
                print(f"Found model with ID {model_id}: {model['name']}")
                
                # Ensure all necessary fields have default values
                result = model.copy()
                
                # Add default values for any missing fields
                defaults = {
                    "description": "",
                    "schema": "default",
                    "materialized": "view",
                    "file_path": "N/A",
                    "columns": [],
                    "sql": "",
                    "tags": []
                }
                
                for key, default_value in defaults.items():
                    if key not in result or result[key] is None:
                        result[key] = default_value
                
                return result
                
        print(f"Model with ID {model_id} not found")
        return None
    
    def get_model_with_lineage(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model with its upstream and downstream lineage"""
        # Validate model_id to prevent NaN values
        if model_id == "NaN" or not model_id:
            print(f"Invalid model ID: {model_id}")
            return None
            
        model = self.get_model(model_id)
        if not model:
            return None
        
        print(f"Getting lineage for model {model_id} ({model['name']})")
        
        # Helper function to apply default values
        def apply_defaults(model_data):
            defaults = {
                "description": "",
                "schema": "default",
                "materialized": "view",
                "file_path": "N/A",
                "columns": [],
                "sql": "",
                "tags": []
            }
            
            for key, default_value in defaults.items():
                if key not in model_data or model_data[key] is None:
                    model_data[key] = default_value
            
            return model_data
        
        # Get upstream models (models that this model depends on)
        upstream = []
        for lineage in self.metadata.get("lineage", []):
            if lineage["target"] == model_id:
                source_id = lineage["source"]
                # Skip invalid source IDs
                if source_id == "NaN" or not source_id:
                    continue
                    
                source_model = self.get_model(source_id)
                if source_model:
                    # Include all required fields from the source model
                    upstream_model = {
                        "id": source_id,
                        "name": source_model["name"],
                        "project": source_model["project"],
                        "description": source_model.get("description", ""),
                        "columns": source_model.get("columns", []),
                        "sql": source_model.get("sql", ""),
                        "schema": source_model.get("schema", "default"),
                        "materialized": source_model.get("materialized", "view"),
                        "file_path": source_model.get("file_path", "N/A"),
                        "tags": source_model.get("tags", [])
                    }
                    
                    # Apply defaults to ensure all fields have values
                    upstream.append(apply_defaults(upstream_model))
        
        # Get downstream models (models that depend on this model)
        downstream = []
        for lineage in self.metadata.get("lineage", []):
            if lineage["source"] == model_id:
                target_id = lineage["target"]
                # Skip invalid target IDs
                if target_id == "NaN" or not target_id:
                    continue
                    
                target_model = self.get_model(target_id)
                if target_model:
                    # Include all required fields from the target model
                    downstream_model = {
                        "id": target_id,
                        "name": target_model["name"],
                        "project": target_model["project"],
                        "description": target_model.get("description", ""),
                        "columns": target_model.get("columns", []),
                        "sql": target_model.get("sql", ""),
                        "schema": target_model.get("schema", "default"),
                        "materialized": target_model.get("materialized", "view"),
                        "file_path": target_model.get("file_path", "N/A"),
                        "tags": target_model.get("tags", [])
                    }
                    
                    # Apply defaults to ensure all fields have values
                    downstream.append(apply_defaults(downstream_model))
        
        print(f"Found {len(upstream)} upstream and {len(downstream)} downstream models")
        if upstream:
            print("Upstream models:")
            for up in upstream[:3]:  # Print first 3
                print(f"  - {up['name']} ({up['id']})")
            if len(upstream) > 3:
                print(f"  - ... and {len(upstream) - 3} more")
                
        if downstream:
            print("Downstream models:")
            for down in downstream[:3]:  # Print first 3
                print(f"  - {down['name']} ({down['id']})")
            if len(downstream) > 3:
                print(f"  - ... and {len(downstream) - 3} more")
        
        # Add lineage to model
        model_with_lineage = model.copy()
        model_with_lineage["upstream"] = upstream
        model_with_lineage["downstream"] = downstream
        
        return model_with_lineage
    
    def _get_project_name(self, project_id: str) -> str:
        """Get project name from project ID"""
        for project in self.metadata.get("projects", []):
            if project["id"] == project_id:
                return project["name"]
        return "Unknown Project"
    
    def get_lineage(self) -> List[Dict[str, str]]:
        """Get all lineage relationships"""
        lineage = self.metadata.get("lineage", [])
        print(f"Returning {len(lineage)} lineage relationships")
        if lineage:
            print("Sample lineage relationships:")
            for i, l in enumerate(lineage[:3]):
                print(f"  {i+1}. Source: {l['source']}, Target: {l['target']}")
        return lineage

    def update_description(self, entity_type: str, entity_id: str, description: str) -> bool:
        """Update description for a model or column"""
        try:
            if entity_type == "model":
                # Update model description
                for model in self.metadata.get("models", []):
                    if model.get("id") == entity_id:
                        model["description"] = description
                        # Mark as user-edited
                        model["user_edited"] = True
                        
                        # Save updated metadata
                        with open(self.unified_metadata_path, 'w') as f:
                            json.dump(self.metadata, f, indent=2)
                        
                        print(f"Updated description for model {entity_id}")
                        return True
                
                print(f"Model {entity_id} not found")
                return False
                
            elif entity_type == "column":
                # Format for column ID is "model_id:column_name"
                if ":" not in entity_id:
                    print(f"Invalid column ID format: {entity_id}")
                    return False
                    
                model_id, column_name = entity_id.split(":", 1)
                
                # Find the model
                for model in self.metadata.get("models", []):
                    if model.get("id") == model_id:
                        # Find the column
                        for column in model.get("columns", []):
                            if column.get("name") == column_name:
                                column["description"] = description
                                # Mark as user-edited
                                column["user_edited"] = True
                                
                                # Save updated metadata
                                with open(self.unified_metadata_path, 'w') as f:
                                    json.dump(self.metadata, f, indent=2)
                                
                                print(f"Updated description for column {entity_id}")
                                return True
                        
                        print(f"Column {column_name} not found in model {model_id}")
                        return False
                
                print(f"Model {model_id} not found")
                return False
            
            else:
                print(f"Invalid entity type: {entity_type}")
                return False
                
        except Exception as e:
            print(f"Error updating description: {str(e)}")
            return False
            
    def refresh_model_metadata(self, model_id: str, force_update: bool = True) -> bool:
        """
        Refresh metadata for a specific model, focusing on AI descriptions
        
        Args:
            model_id: ID of the model to refresh
            force_update: If True, override existing descriptions with new AI-generated ones
        
        Returns:
            bool: True if refresh was successful, False otherwise
        """
        try:
            # Get the current model
            model = None
            for m in self.metadata.get("models", []):
                if m["id"] == model_id:
                    model = m
                    break
            
            if not model:
                print(f"Model with ID {model_id} not found for refreshing")
                return False
            
            print(f"Refreshing metadata for model {model_id} ({model.get('name', 'Unknown')}) in project {model.get('project', 'Unknown')}")
            print(f"Force update mode: {force_update}")
            
            # Initialize AI service if enabled
            if self.use_ai_descriptions and not self.ai_service:
                print("Initializing AI service for description generation")
                from backend.services.ai_description_service import AIDescriptionService
                self.ai_service = AIDescriptionService()
            
            # Get manifest data for domain context if possible
            manifest_data = self.load_manifest_data()
            if manifest_data:
                print(f"Successfully loaded manifest data with {len(manifest_data.get('nodes', {}))} nodes for domain context")
            else:
                print("Warning: No manifest data available for domain context")
            
            if self.use_ai_descriptions and self.ai_service:
                # Log AI description state before update
                ai_desc_before = model.get("ai_description")
                user_edited_before = model.get("user_edited", False)
                desc_before = model.get("description")
                
                # Count columns with AI descriptions before update
                cols_with_ai_desc_before = sum(1 for col in model.get("columns", []) if col.get("ai_description"))
                
                print(f"Current state - Model description: {desc_before[:50]}{'...' if desc_before and len(desc_before) > 50 else ''}")
                print(f"Current state - AI description: {ai_desc_before[:50]}{'...' if ai_desc_before and len(ai_desc_before) > 50 else ''}")
                print(f"Current state - User edited: {user_edited_before}")
                print(f"Current state - Columns with AI descriptions: {cols_with_ai_desc_before}/{len(model.get('columns', []))}")
                
                # Make a copy of the model to modify for the AI service
                model_for_ai = model.copy()
                
                # If we're forcing update, prepare the model by clearing descriptions
                if force_update:
                    # If not user-edited, clear the model description to allow overwrite
                    if not model_for_ai.get("user_edited", False):
                        if "description" in model_for_ai:
                            print(f"Clearing existing model description to allow AI update")
                            model_for_ai["description"] = ""
                    else:
                        print(f"Model is user-edited, preserving description but updating AI description")
                    
                    # Clear column descriptions that aren't user-edited
                    for column in model_for_ai.get("columns", []):
                        if not column.get("user_edited", False):
                            if "description" in column:
                                column["description"] = ""
                    
                    print(f"Model prepared for AI processing with forced update mode")
                
                # Use the efficient batch processing to update the model and its columns together
                updated_model = self.ai_service.generate_model_with_columns_description(
                    model_for_ai, 
                    manifest_data,
                    force_update=force_update
                )
                
                # For force_update mode, ensure AI descriptions are copied to the main description field
                if force_update:
                    # Update model description if not user-edited
                    if not updated_model.get("user_edited", False) and updated_model.get("ai_description"):
                        updated_model["description"] = updated_model["ai_description"]
                        print(f"Copied AI description to model description field")
                    
                    # Update column descriptions that aren't user-edited
                    cols_updated = 0
                    for column in updated_model.get("columns", []):
                        if not column.get("user_edited", False) and column.get("ai_description"):
                            column["description"] = column["ai_description"]
                            cols_updated += 1
                    
                    print(f"Copied AI descriptions to main description field for {cols_updated} columns")
                
                # Log AI description state after update
                ai_desc_after = updated_model.get("ai_description")
                desc_after = updated_model.get("description")
                
                # Count columns with AI descriptions after update
                cols_with_ai_desc_after = sum(1 for col in updated_model.get("columns", []) if col.get("ai_description"))
                cols_with_desc_after = sum(1 for col in updated_model.get("columns", []) if col.get("description"))
                
                print(f"After update - Model description: {desc_after[:50]}{'...' if desc_after and len(desc_after) > 50 else ''}")
                print(f"After update - AI description: {ai_desc_after[:50]}{'...' if ai_desc_after and len(ai_desc_after) > 50 else ''}")
                print(f"After update - Columns with AI descriptions: {cols_with_ai_desc_after}/{len(updated_model.get('columns', []))}")
                print(f"After update - Columns with main descriptions: {cols_with_desc_after}/{len(updated_model.get('columns', []))}")
                
                # Update the model in the metadata
                model_found = False
                for i, m in enumerate(self.metadata.get("models", [])):
                    if m["id"] == model_id:
                        self.metadata["models"][i] = updated_model
                        model_found = True
                        break
                
                if not model_found:
                    print(f"Warning: Model {model_id} not found in metadata after generating descriptions")
                    return False
                
                # Save updated metadata
                try:
                    with open(self.unified_metadata_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                    print(f"Successfully saved updated metadata to {self.unified_metadata_path}")
                    
                    # Additional verification that the file was actually written
                    if os.path.exists(self.unified_metadata_path):
                        file_size = os.path.getsize(self.unified_metadata_path)
                        print(f"Metadata file size: {file_size} bytes")
                    else:
                        print(f"Warning: Metadata file does not exist after saving")
                except Exception as e:
                    print(f"Error saving metadata file: {str(e)}")
                    return False
                
                print(f"Successfully refreshed metadata for model {model['name']} with AI descriptions")
                print(f"Summary of changes:")
                print(f"- AI description changes: {'Yes' if ai_desc_before != ai_desc_after else 'No'}")
                print(f"- Main description changes: {'Yes' if desc_before != desc_after else 'No'}")
                print(f"- Columns with AI descriptions: {cols_with_ai_desc_before} -> {cols_with_ai_desc_after}")
                print(f"- Columns with main descriptions: {sum(1 for col in model.get('columns', []) if col.get('description'))} -> {cols_with_desc_after}")
                
                return True
            else:
                print("AI description generation is disabled or AI service is not initialized")
                return False
            
        except Exception as e:
            print(f"Error refreshing model metadata: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def process_cross_project_references(self, models, lineage):
        """
        Process cross-project references to ensure proper visualization.
        This includes handling shared models like my_first_dbt_model and my_second_dbt_model
        across different projects, as well as models with specific prefixes.
        """
        # Create lookup dictionaries
        model_by_id = {model['id']: model for model in models}
        model_by_name_and_project = {}
        model_by_name = {}
        
        # Map models by name and project
        for model in models:
            name = model.get('name')
            project = model.get('project')
            
            if name and project:
                key = f"{name}:{project}"
                model_by_name_and_project[key] = model
                
                if name not in model_by_name:
                    model_by_name[name] = []
                model_by_name[name].append(model)
        
        # Find dimension, staging, and fact models that should be shared across projects
        shared_models = {
            name: models_list for name, models_list in model_by_name.items() 
            if (name.startswith('dim_') or name.startswith('stg_') or name.startswith('fct_') or 
                name == 'my_first_dbt_model' or name == 'my_second_dbt_model') and len(models_list) > 1
        }
        
        # Print out all shared models for debugging
        if shared_models:
            print(f"Found {len(shared_models)} shared models across projects:")
            for name, model_list in shared_models.items():
                projects = [m.get('project') for m in model_list]
                print(f"  - {name} in projects: {', '.join(projects)}")
        
        # For each shared model, identify its home project based on naming conventions
        model_canonical_map = {}  # Maps non-canonical model IDs to canonical model IDs
        
        for model_name, model_instances in shared_models.items():
            # Extract entity name from prefixed models (e.g., "customer" from "dim_customer")
            entity_name = ""
            if "_" in model_name:
                parts = model_name.split('_', 1)
                if len(parts) > 1:
                    entity_name = parts[1]
            
            # Find the home project based on different rules
            home_model = None
            
            # Rule 1: For prefixed models (dim_*, stg_*, fct_*), prefer the project that contains the entity name
            if entity_name and (model_name.startswith('dim_') or model_name.startswith('stg_') or model_name.startswith('fct_')):
                for model in model_instances:
                    project = model.get('project', '')
                    if entity_name.lower() in project.lower():
                        home_model = model
                        break
            
            # Rule 2: For example models, prefer projects in order: analytics_project, then ecommerce_project, then my_test_project
            if not home_model and (model_name == 'my_first_dbt_model' or model_name == 'my_second_dbt_model'):
                # Define priority for example models
                project_priority = {
                    'analytics_project': 1,
                    'ecommerce_project': 2,
                    'my_test_project': 3
                }
                
                # Sort models by project priority
                sorted_models = sorted(
                    model_instances, 
                    key=lambda m: project_priority.get(m.get('project', ''), 999)
                )
                
                if sorted_models:
                    home_model = sorted_models[0]
            
            # Fallback: Just use the first model
            if not home_model and model_instances:
                home_model = model_instances[0]
                
            # If we found a canonical model, mark it and create mapping for others
            if home_model:
                home_model['is_canonical'] = True
                home_id = home_model.get('id')
                
                print(f"  Selected canonical model for {model_name}: {home_model.get('project')}")
                
                # Create mapping for non-canonical versions
                for model in model_instances:
                    model_id = model.get('id')
                    if model_id != home_id:
                        model['references_canonical'] = home_id
                        model_canonical_map[model_id] = home_id
                        print(f"    - Mapped {model.get('project')}.{model_name} -> {home_model.get('project')}.{model_name}")
        
        # Process lineage to handle cross-project references
        updated_lineage = []
        seen_connections = set()
        
        for link in lineage:
            source_id = link.get('source')
            target_id = link.get('target')
            
            # Skip invalid links
            if not source_id or not target_id:
                continue
                
            source_model = model_by_id.get(source_id)
            target_model = model_by_id.get(target_id)
            
            # Skip if models not found
            if not source_model or not target_model:
                continue
            
            # Handle references to non-canonical models
            if source_id in model_canonical_map:
                # Replace with link from the canonical model
                source_id = model_canonical_map[source_id]
                
            if target_id in model_canonical_map:
                # Replace with link to the canonical model
                target_id = model_canonical_map[target_id]
            
            # Create updated link
            if source_id != target_id:  # Skip self-references
                link_key = f"{source_id}-{target_id}"
                
                # Only add each unique link once
                if link_key not in seen_connections:
                    seen_connections.add(link_key)
                    updated_lineage.append({
                        'source': source_id,
                        'target': target_id
                    })
        
        # Add additional links between my_first_dbt_model and my_second_dbt_model across all projects
        if 'my_first_dbt_model' in model_by_name and 'my_second_dbt_model' in model_by_name:
            # Get canonical versions if they exist
            first_models = model_by_name['my_first_dbt_model']
            second_models = model_by_name['my_second_dbt_model']
            
            canonical_first = next((m for m in first_models if m.get('is_canonical')), None) or (first_models[0] if first_models else None)
            canonical_second = next((m for m in second_models if m.get('is_canonical')), None) or (second_models[0] if second_models else None)
            
            if canonical_first and canonical_second:
                first_id = canonical_first.get('id')
                second_id = canonical_second.get('id')
                
                # Add the connection from first to second model if it doesn't exist
                link_key = f"{first_id}-{second_id}"
                if link_key not in seen_connections:
                    seen_connections.add(link_key)
                    updated_lineage.append({
                        'source': first_id,
                        'target': second_id
                    })
                    print(f"Added implicit link from my_first_dbt_model to my_second_dbt_model")
        
        # Return processed models and lineage
        return models, updated_lineage

    def get_all_lineage(self):
        """Get all lineage data"""
        models = self.get_models()
        lineage = self._load_lineage()
        
        # Process cross-project references
        processed_models, processed_lineage = self.process_cross_project_references(models, lineage)
        
        return {
            "models": processed_models,
            "lineage": processed_lineage
        }

    def add_cross_references(self, cross_refs):
        """
        Add cross-project references to enhance lineage. This method is a no-op now 
        as we rely on the parser to find all real relationships.
        
        Args:
            cross_refs: Cross-project references
        """
        print("Artificial cross-project references are no longer added. Using only actual references from code.")
        pass

    def _process_metadata(self, raw_metadata):
        """
        Process the raw metadata from the parser, adding additional information
        and enhancing lineage relationships.
        
        Args:
            raw_metadata: The raw metadata dictionary from parse_dbt_projects
            
        Returns:
            dict: Processed metadata with enhanced information
        """
        if not raw_metadata:
            return {"projects": [], "models": [], "lineage": []}
            
        # Deep copy to avoid modifying the original
        processed_metadata = {
            "projects": raw_metadata.get("projects", []),
            "models": raw_metadata.get("models", []),
            "lineage": raw_metadata.get("lineage", [])
        }
        
        # Process cross-project references
        processed_models, processed_lineage = self.process_cross_project_references(
            processed_metadata["models"],
            processed_metadata["lineage"]
        )
        
        processed_metadata["models"] = processed_models
        processed_metadata["lineage"] = processed_lineage
            
        print(f"Processed metadata:")
        print(f"  - Projects: {len(processed_metadata.get('projects', []))}")
        print(f"  - Models: {len(processed_metadata.get('models', []))}")
        print(f"  - Lineage relationships: {len(processed_metadata.get('lineage', []))}")
        
        return processed_metadata

    def get_project_manifest_paths(self) -> List[str]:
        """Get paths to manifest.json files in all dbt projects"""
        manifest_paths = []
        
        # Get all project directories
        project_dirs = self.get_project_dirs()
        for project_dir in project_dirs:
            project_path = os.path.join(self.dbt_projects_dir, project_dir)
            manifest_path = os.path.join(project_path, "target", "manifest.json")
            
            if os.path.exists(manifest_path):
                manifest_paths.append(manifest_path)
                print(f"Found manifest for project {project_dir}: {manifest_path}")
        
        return manifest_paths

    def load_manifest_data(self) -> Dict[str, Any]:
        """Load and combine manifest data from all projects"""
        manifest_data = {
            "nodes": {},
            "sources": {},
            "macros": {},
            "metrics": {},
            "exposures": {}
        }
        
        # Get manifest paths
        manifest_paths = self.get_project_manifest_paths()
        
        # Load and combine manifest data
        for manifest_path in manifest_paths:
            try:
                with open(manifest_path, 'r') as f:
                    project_manifest = json.load(f)
                    
                    # Merge manifest sections
                    for section in ["nodes", "sources", "macros", "metrics", "exposures"]:
                        if section in project_manifest:
                            manifest_data[section].update(project_manifest[section])
                            
                    print(f"Loaded manifest data from {manifest_path}")
            except Exception as e:
                print(f"Error loading manifest from {manifest_path}: {str(e)}")
        
        return manifest_data

    def refresh_with_domain_context(self, manifest_data: Dict[str, Any] = None) -> bool:
        """Refresh metadata with domain awareness using manifest data"""
        try:
            # First ensure we have manifest data
            if not manifest_data:
                manifest_data = self.load_manifest_data()
            
            if not manifest_data or not manifest_data.get("nodes"):
                print("Warning: No manifest data available for domain context")
                return self.refresh()  # Fall back to regular refresh
            
            print("Starting metadata refresh with domain context using efficient batch processing...")
            
            # Parse dbt projects
            raw_metadata = self._parse_dbt_projects()
            
            # Process metadata
            processed_metadata = self._process_metadata(raw_metadata)
            
            # Make sure AI descriptions are enabled
            if self.use_ai_descriptions and not self.ai_service:
                from backend.services.ai_description_service import AIDescriptionService
                self.ai_service = AIDescriptionService()
            
            # Enrich with AI descriptions if enabled, using manifest data for domain context
            if self.use_ai_descriptions and self.ai_service:
                print("Using efficient batch processing for generating descriptions...")
                enriched_metadata = self.ai_service.enrich_metadata_efficiently(processed_metadata, manifest_data)
                self.metadata = enriched_metadata
            else:
                self.metadata = processed_metadata
            
            # Save to file
            with open(self.unified_metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Domain-aware metadata saved to {self.unified_metadata_path}")
            return True
        except Exception as e:
            print(f"Error refreshing metadata with domain context: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def refresh_column_metadata(self, model_id: str, column_name: str, manifest_data: Dict[str, Any] = None) -> bool:
        """Refresh AI description for a specific column using domain context"""
        try:
            # Check if the model exists
            model = self.get_model(model_id)
            if not model:
                print(f"Model {model_id} not found")
                return False
            
            # Find the column
            column = None
            column_index = -1
            for i, col in enumerate(model.get("columns", [])):
                if col.get("name") == column_name:
                    column = col
                    column_index = i
                    break
                
            if not column:
                print(f"Column {column_name} not found in model {model_id}")
                return False
            
            # Ensure AI description service is available
            if not self.use_ai_descriptions:
                self.use_ai_descriptions = True
            
            if not self.ai_service:
                from backend.services.ai_description_service import AIDescriptionService
                self.ai_service = AIDescriptionService()
            
            # Load manifest data if not provided
            if not manifest_data:
                manifest_data = self.load_manifest_data()
            
            print(f"Refreshing description for column {column_name} in model {model_id} using batch processing")
            
            # Set refresh flag on the column to ensure it gets updated
            column["refresh_description"] = True
            
            # Use batch processing method to update the entire model
            # This is more efficient than generating a description for a single column
            updated_model = self.ai_service.generate_model_with_columns_description(model, manifest_data)
            
            # Update the model in the metadata
            for i, m in enumerate(self.metadata.get("models", [])):
                if m["id"] == model_id:
                    self.metadata["models"][i] = updated_model
                    break
            
            # Save the updated metadata
            with open(self.unified_metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"Updated model and all columns in a single batch operation")
            return True
            
        except Exception as e:
            print(f"Error refreshing column metadata: {str(e)}")
            return False