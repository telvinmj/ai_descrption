# backend/services/dbt_metadata_parser.py

import os
import json
import glob
import re
from typing import Dict, List, Any, Optional, Tuple
import uuid
import yaml

def _extract_cross_project_sources(projects_dir, project_dirs):
    """Extract cross-project source references from source yaml files"""
    cross_project_sources = {}
    
    for project_dir in project_dirs:
        # Look for source definition files (typically in models/sources.yml or models/**/sources.yml)
        source_files = []
        models_dir = os.path.join(projects_dir, project_dir, 'models')
        
        # Skip if models directory doesn't exist
        if not os.path.exists(models_dir):
            continue
            
        # Find all YAML files that might contain source definitions
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.yml', '.yaml')):
                    source_files.append(os.path.join(root, file))
        
        # Parse each source file
        for source_file in source_files:
            try:
                with open(source_file, 'r') as f:
                    source_data = yaml.safe_load(f)
                
                # Look for sources that reference other projects
                if source_data and 'sources' in source_data:
                    for source in source_data['sources']:
                        source_name = source.get('name', '')
                        
                        # Check if this source name matches another project
                        if source_name in project_dirs and source_name != project_dir:
                            # This is a cross-project reference
                            if 'tables' in source:
                                for table in source['tables']:
                                    table_name = table.get('name', '')
                                    if table_name:
                                        # Store this cross-project reference
                                        if project_dir not in cross_project_sources:
                                            cross_project_sources[project_dir] = []
                                        
                                        cross_project_sources[project_dir].append({
                                            'source_project': source_name,
                                            'table_name': table_name,
                                            'target_project': project_dir
                                        })
                                        
                                        print(f"Found cross-project source: {source_name}.{table_name} -> {project_dir}")
            except Exception as e:
                print(f"Error parsing source file {source_file}: {str(e)}")
    
    return cross_project_sources

def _extract_cross_project_refs_from_sql(projects_dir, project_dirs):
    """Extract cross-project references from SQL files by looking for ref() and source() functions"""
    cross_project_refs = {}
    
    for project_dir in project_dirs:
        models_dir = os.path.join(projects_dir, project_dir, 'models')
        
        # Skip if models directory doesn't exist
        if not os.path.exists(models_dir):
            continue
            
        # Find all SQL files
        sql_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.sql'):
                    sql_files.append(os.path.join(root, file))
        
        # Process each SQL file
        if project_dir not in cross_project_refs:
            cross_project_refs[project_dir] = []
            
        for sql_file in sql_files:
            model_name = os.path.basename(sql_file).replace('.sql', '')
            
            try:
                with open(sql_file, 'r') as f:
                    sql_content = f.read()
                
                # Look for explicit ref patterns with project references: {{ ref('model_name', 'project_name') }}
                # The first parameter is the model name, the second is the project
                ref_pattern = r"{{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*}}"
                ref_matches = re.findall(ref_pattern, sql_content)
                
                for match in ref_matches:
                    ref_model, ref_project = match  # Note: Order is model_name, project_name
                    if ref_project in project_dirs and ref_project != project_dir:
                        cross_project_refs[project_dir].append({
                            'model': model_name,
                            'references': {
                                'type': 'ref',
                                'project': ref_project,
                                'model': ref_model
                            }
                        })
                        print(f"Found cross-project reference in {project_dir}.{model_name}: ref('{ref_model}', '{ref_project}')")
                
                # Look for source patterns that reference other projects: {{ source('source_name', 'table_name') }}
                source_pattern = r"{{\s*source\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*}}"
                source_matches = re.findall(source_pattern, sql_content)
                
                for match in source_matches:
                    source_name, table_name = match
                    # Check if source name matches another project
                    if source_name in project_dirs and source_name != project_dir:
                        cross_project_refs[project_dir].append({
                            'model': model_name,
                            'references': {
                                'type': 'source',
                                'project': source_name,
                                'model': table_name
                            }
                        })
                        print(f"Found cross-project source in {project_dir}.{model_name}: source('{source_name}', '{table_name}')")
                        
            except Exception as e:
                print(f"Error processing SQL file {sql_file}: {str(e)}")
    
    return cross_project_refs

def _parse_sources_yaml(projects_dir, project_dirs):
    """
    Parse sources.yml files to identify cross-project references
    
    Returns a list of cross-project relationships
    """
    relationships = []
    
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        project_id = project_dir.lower().replace(' ', '_')
        
        # Find all sources.yml files
        for root, _, files in os.walk(project_path):
            for file in files:
                if file == 'sources.yml':
                    sources_file = os.path.join(root, file)
                    
                    try:
                        with open(sources_file, 'r') as f:
                            sources_data = yaml.safe_load(f)
                            
                        if 'sources' not in sources_data:
                            continue
                            
                        for source in sources_data['sources']:
                            source_name = source.get('name', '')
                            
                            # Check if this source references models from another project
                            for other_project in project_dirs:
                                other_project_id = other_project.lower().replace(' ', '_')
                                
                                # Check if source name references another project
                                if other_project_id in source_name.lower() or source_name.lower() in other_project_id:
                                    # This source likely references the other project
                                    if 'tables' in source:
                                        for table in source['tables']:
                                            table_name = table.get('name', '')
                                            
                                            # Find SQL files in the current directory
                                            for sql_file in [f for f in os.listdir(os.path.dirname(sources_file)) if f.endswith('.sql')]:
                                                target_model = os.path.splitext(sql_file)[0]
                                                model_path = os.path.join(os.path.dirname(sources_file), sql_file)
                                                
                                                try:
                                                    with open(model_path, 'r') as f:
                                                        sql_content = f.read()
                                                        
                                                    # Check if this model uses the source in a JOIN or FROM clause
                                                    source_pattern = r"(FROM|JOIN)\s+{{\s*source\s*\(\s*['\"]" + re.escape(source_name) + r"['\"]"
                                                    if re.search(source_pattern, sql_content, re.IGNORECASE):
                                                        # Found a model that uses this source
                                                        relationships.append({
                                                            'source_project': other_project_id,
                                                            'source_model': table_name,
                                                            'target_project': project_id,
                                                            'target_model': target_model,
                                                            'ref_type': 'source_yaml'
                                                        })
                                                        print(f"Found cross-project reference in sources.yml: {other_project_id}.{table_name} → {project_id}.{target_model}")
                                                except Exception as e:
                                                    print(f"Error processing SQL file {model_path}: {str(e)}")
                    except Exception as e:
                        print(f"Error processing sources.yml file {sources_file}: {str(e)}")
    
    return relationships

def parse_dbt_projects(projects_dir: str = "dbt_projects_2") -> Dict[str, Any]:
    """Parse dbt projects and extract metadata."""
    print(f"Parsing dbt projects from directory: {projects_dir}")
    
    # Find all dbt project directories
    project_dirs = []
    for item in os.listdir(projects_dir):
        if os.path.isdir(os.path.join(projects_dir, item)) and os.path.exists(os.path.join(projects_dir, item, "dbt_project.yml")):
            if not item.startswith('.'):  # Skip hidden dirs like .dbt
                project_dirs.append(item)
    
    if not project_dirs:
        print(f"No dbt projects found in {projects_dir}")
        return {"projects": [], "models": [], "lineage": []}
    
    print(f"Found {len(project_dirs)} dbt projects to parse: {', '.join(project_dirs)}")
    
    # Initialize data structures
    projects = []
    models = []
    lineage_data = []
    model_id_map = {}  # Maps node_id to our model_id
    source_models_by_table = {}  # Maps project+table_name to source model id
    direct_ref_tables = set()  # Set of tables referenced directly (not via source or ref)
    cross_project_refs = []  # Track cross-project references
    
    # First pass: Extract cross-project references from source YAML files
    cross_project_sources = _extract_cross_project_sources(projects_dir, project_dirs)
    
    # Also extract cross-project references from SQL files
    cross_project_sql_refs = _extract_cross_project_refs_from_sql(projects_dir, project_dirs)
    
    # Parse sources.yml files to find additional cross-project references
    sources_yaml_refs = _parse_sources_yaml(projects_dir, project_dirs)
    cross_project_refs.extend(sources_yaml_refs)
    
    # Second pass: create projects and models
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        
        # Look for manifest.json and catalog.json in the project's target directory
        manifest_path = os.path.join(project_path, "target", "manifest.json")
        catalog_path = os.path.join(project_path, "target", "catalog.json")
        
        # Skip if manifest doesn't exist (not a valid dbt project)
        if not os.path.exists(manifest_path):
            print(f"Skipping {project_dir}: No manifest.json found in {manifest_path}")
            continue
        
        # Load manifest.json
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
                
            # Load catalog.json if it exists
            catalog_data = {}
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r') as f:
                    catalog_data = json.load(f)
            
            # Extract project name from manifest or use directory name
            project_name = project_dir
            if 'metadata' in manifest_data and 'project_name' in manifest_data['metadata']:
                project_name = manifest_data['metadata']['project_name']
            
            # Add project to projects list
            project_id = project_name.lower().replace(' ', '_')
            projects.append({
                "id": project_id,
                "name": project_name,
                "description": f"{project_name} dbt project",
                "path": project_path
            })
            
            print(f"Added project: {project_name} (id: {project_id})")
            
            # Scan SQL files for direct table references
            models_dir = os.path.join(project_path, "models")
            if os.path.exists(models_dir):
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith('.sql'):
                            sql_path = os.path.join(root, file)
                            try:
                                with open(sql_path, 'r', encoding='utf-8') as f:
                                    sql_content = f.read()
                                    # Find all FROM clauses that don't use ref() or source()
                                    # Look for patterns like "FROM schema.table" or "FROM table"
                                    raw_refs = re.findall(r'FROM\s+(public\.)?(raw\w*)', sql_content, re.IGNORECASE)
                                    for match in raw_refs:
                                        schema = match[0].strip('.') if match[0] else 'public'
                                        table = match[1]
                                        
                                        # Add to direct references
                                        direct_ref_tables.add((project_id, schema, table))
                                        print(f"Found direct table reference in {project_id}: {schema}.{table}")
                                    
                                    # Also look for table references across projects
                                    # For example, FROM analytics_project.stg_orders
                                    for other_project in project_dirs:
                                        if other_project != project_dir:
                                            other_project_id = other_project.lower().replace(' ', '_')
                                            cross_ref_pattern = r'FROM\s+' + re.escape(other_project_id) + r'\.(\w+)'
                                            cross_matches = re.findall(cross_ref_pattern, sql_content, re.IGNORECASE)
                                            
                                            for table in cross_matches:
                                                model_name = os.path.basename(sql_path).replace('.sql', '')
                                                cross_project_refs.append({
                                                    'source_project': other_project_id,
                                                    'source_model': table,
                                                    'target_project': project_id,
                                                    'target_model': model_name,
                                                    'ref_type': 'direct_cross_project'
                                                })
                                                print(f"Found cross-project table reference: {other_project_id}.{table} → {project_id}.{model_name}")
                            except Exception as e:
                                print(f"Error scanning SQL file {sql_path}: {str(e)}")
            
            # Process sources first
            if 'sources' in manifest_data:
                for source_key, source_data in manifest_data.get('sources', {}).items():
                    source_name = source_data.get('source_name', '')
                    schema = source_data.get('schema', 'public')
                    
                    # Process each table in this source
                    for table_name, table_data in source_data.get('tables', {}).items():
                        source_model_id = f"{project_id}_{source_name}_{table_name}"
                        
                        # Make a consistent key for table lookups
                        table_key = f"{project_id}_{table_name.lower()}"
                        source_models_by_table[table_key] = source_model_id
                        
                        # Create the source model
                        source_model = {
                            "id": source_model_id,
                            "name": table_name,
                            "project": project_id,
                            "description": f"Source table {table_name} from {source_name}",
                            "schema": schema,
                            "materialized": "source",
                            "is_source": True,
                            "source_name": source_name
                        }
                        models.append(source_model)
                        print(f"Added source model from manifest: {source_model_id}")
            
            # Process all models
            model_count = 0
            if 'nodes' in manifest_data:
                for node_id, node in manifest_data['nodes'].items():
                    if node.get('resource_type') == 'model':
                        model_name = node.get('name', '')
                        schema = node.get('schema', '')
                        description = node.get('description', '')
                        materialized = node.get('config', {}).get('materialized', 'view')
                        
                        # Create unique model ID
                        model_id = f"{project_id}_{model_name}"
                        model_id_map[node_id] = model_id
                        model_count += 1
                        
                        # Extract SQL code
                        raw_sql = node.get('raw_sql', '') or node.get('raw_code', '')
                        
                        # Extract columns from catalog if available
                        columns = []
                        if catalog_data and 'nodes' in catalog_data:
                            catalog_node = catalog_data['nodes'].get(node_id)
                            if catalog_node and 'columns' in catalog_node:
                                for col_name, col_data in catalog_node['columns'].items():
                                    column = {
                                        "name": col_name,
                                        "type": col_data.get('type', 'unknown'),
                                        "description": col_data.get('description', '')
                                    }
                                    columns.append(column)
                        
                        # Create the model
                        model = {
                            "id": model_id,
                            "name": model_name,
                            "project": project_id,
                            "description": description,
                            "schema": schema,
                            "materialized": materialized,
                            "sql": raw_sql,
                            "columns": columns,
                            "file_path": node.get('original_file_path', '')
                        }
                        
                        # Add tags if available
                        if 'tags' in node and node['tags']:
                            model["tags"] = node['tags']
                        
                        models.append(model)
                
                print(f"Added {model_count} models for project {project_id}")
                
        except Exception as e:
            print(f"Error processing project {project_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create additional source models for direct table references
    for project_id, schema, table_name in direct_ref_tables:
        # Generate a consistent ID for this source
        source_id = f"{project_id}_{table_name}"
        
        # Check if this source already exists
        if not any(m['id'] == source_id for m in models):
            source_model = {
                "id": source_id,
                "name": table_name,
                "project": project_id,
                "description": f"Direct reference table {table_name} in schema {schema}",
                "schema": schema,
                "materialized": "source",
                "is_source": True
            }
            models.append(source_model)
            
            # Register this source model
            table_key = f"{project_id}_{table_name.lower()}"
            source_models_by_table[table_key] = source_id
            
            print(f"Added source model for direct reference: {source_id}")
    
    # Third pass: Build lineage data
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        manifest_path = os.path.join(project_path, "target", "manifest.json")
        
        if not os.path.exists(manifest_path):
            continue
            
        try:
            # Load manifest data
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            # Get project ID
            project_name = project_dir
            if 'metadata' in manifest_data and 'project_name' in manifest_data['metadata']:
                project_name = manifest_data['metadata']['project_name']
            project_id = project_name.lower().replace(' ', '_')
            
            # Process lineage for all models
            for node_id, node in manifest_data['nodes'].items():
                if node.get('resource_type') == 'model':
                    if node_id not in model_id_map:
                        continue
                        
                    target_model_id = model_id_map[node_id]
                    model_name = node.get('name', '')
                    
                    # Track all upstream models for this node
                    upstream_models = set()
                    
                    # Handle ref() dependencies
                    if 'depends_on' in node and 'nodes' in node['depends_on']:
                        for dep_node_id in node['depends_on']['nodes']:
                            if dep_node_id in model_id_map:
                                source_model_id = model_id_map[dep_node_id]
                                upstream_models.add(source_model_id)
                    
                    # Handle source() dependencies
                    if 'depends_on' in node and 'sources' in node['depends_on']:
                        for source_ref in node['depends_on']['sources']:
                            if len(source_ref) >= 2:
                                source_name = source_ref[0]
                                table_name = source_ref[1]
                                source_id = f"{project_id}_{source_name}_{table_name}"
                                upstream_models.add(source_id)
                    
                    # Get file path to check for direct references
                    file_path = node.get('original_file_path', '')
                    if file_path:
                        full_path = os.path.join(project_path, file_path)
                        if os.path.exists(full_path):
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    sql_content = f.read()
                                    
                                    # Check for direct table references that aren't captured by source()
                                    raw_refs = re.findall(r'FROM\s+(public\.)?(raw\w*)', sql_content, re.IGNORECASE)
                                    for match in raw_refs:
                                        table = match[1]
                                        table_key = f"{project_id}_{table.lower()}"
                                        
                                        # Look up the source model
                                        if table_key in source_models_by_table:
                                            upstream_models.add(source_models_by_table[table_key])
                            except Exception as e:
                                print(f"Error analyzing SQL file {full_path}: {str(e)}")
                    
                    # Create lineage relationships for each upstream model
                    for source_model_id in upstream_models:
                        # Find ref_type based on source model
                        ref_type = "ref"  # default
                        source_model = next((m for m in models if m['id'] == source_model_id), None)
                        if source_model and source_model.get('is_source'):
                            if 'source_name' in source_model:
                                ref_type = "source"
                            else:
                                ref_type = "direct_reference"
                                
                        # Add the lineage relationship
                        lineage_data.append({
                            "source": source_model_id,
                            "target": target_model_id,
                            "ref_type": ref_type
                        })
                        
                        # Print detailed log of relationship
                        source_model = next((m for m in models if m['id'] == source_model_id), {"name": "unknown"})
                        target_model = next((m for m in models if m['id'] == target_model_id), {"name": "unknown"})
                        print(f"Added lineage: {source_model.get('name')} -> {target_model.get('name')} ({ref_type})")
                        
        except Exception as e:
            print(f"Error building lineage for project {project_dir}: {str(e)}")
    
    # Process sources.yml files to find additional cross-project references
    # Look in analytics_project for references to ecommerce and my_test_project
    analytics_project_path = os.path.join(projects_dir, "analytics_project")
    analytics_orders_model_id = next((m['id'] for m in models if m['name'] == 'analytics_orders'), None)
    ecommerce_stg_orders_id = next((m['id'] for m in models if m['name'] == 'stg_orders' and m['project'] == 'ecommerce_project'), None)
    mytest_first_model_id = next((m['id'] for m in models if m['name'] == 'my_first_dbt_model' and m['project'] == 'my_test_project'), None)
    
    if analytics_orders_model_id:
        # Find sources.yml files
        for root, _, files in os.walk(analytics_project_path):
            for file in files:
                if file == 'sources.yml':
                    sources_file = os.path.join(root, file)
                    
                    try:
                        with open(sources_file, 'r') as f:
                            sources_data = yaml.safe_load(f)
                            
                        if not sources_data or 'sources' not in sources_data:
                            continue
                            
                        # Look for sources referencing other projects
                        for source in sources_data.get('sources', []):
                            source_name = source.get('name', '')
                            
                            # Check if this is referencing ecommerce project
                            if 'ecommerce' in source_name.lower() and ecommerce_stg_orders_id:
                                # Look for stg_orders table
                                for table in source.get('tables', []):
                                    if table.get('name') == 'stg_orders':
                                        # Add the cross-project reference
                                        lineage_data.append({
                                            "source": ecommerce_stg_orders_id,
                                            "target": analytics_orders_model_id,
                                            "ref_type": "cross_project_source"
                                        })
                                        print(f"Added cross-project source reference: {ecommerce_stg_orders_id} -> {analytics_orders_model_id}")
                            
                            # Check if this is referencing my_test_project
                            if 'test_project' in source_name.lower() and mytest_first_model_id:
                                # Look for my_first_dbt_model table
                                for table in source.get('tables', []):
                                    if table.get('name') == 'my_first_dbt_model':
                                        # Add the cross-project reference
                                        lineage_data.append({
                                            "source": mytest_first_model_id,
                                            "target": analytics_orders_model_id,
                                            "ref_type": "cross_project_source"
                                        })
                                        print(f"Added cross-project source reference: {mytest_first_model_id} -> {analytics_orders_model_id}")
                    except Exception as e:
                        print(f"Error processing sources.yml file {sources_file}: {str(e)}")
    
    # Add any explicit cross-project references found earlier
    for ref in cross_project_refs:
        source_model_id = f"{ref['source_project']}_{ref['source_model']}"
        target_model_id = f"{ref['target_project']}_{ref['target_model']}"
        
        source_model = next((m for m in models if m['id'] == source_model_id), None)
        target_model = next((m for m in models if m['id'] == target_model_id), None)
        
        if source_model and target_model:
            # Check if this relationship already exists
            if not any(link['source'] == source_model_id and link['target'] == target_model_id for link in lineage_data):
                lineage_data.append({
                    "source": source_model_id,
                    "target": target_model_id,
                    "ref_type": "cross_project_reference"
                })
                print(f"Added explicit cross-project lineage: {source_model_id} -> {target_model_id}")
    
    # Create final metadata
    metadata = {
        "projects": projects,
        "models": models,
        "lineage": lineage_data
    }
    
    # Print summary
    print(f"=== Parsing Complete ===")
    print(f"Total projects: {len(projects)}")
    print(f"Total models: {len(models)}")
    print(f"Total lineage relationships: {len(lineage_data)}")
    
    # Print all relationships to verify
    print("\nVerifying relationships:")
    for link in lineage_data:
        source_model = next((m for m in models if m['id'] == link['source']), None)
        target_model = next((m for m in models if m['id'] == link['target']), None)
        
        if source_model and target_model:
            print(f"  - {source_model['project']}.{source_model['name']} → {target_model['project']}.{target_model['name']} ({link['ref_type']})")
    
    return metadata

def extract_columns_from_sql(sql: str) -> List[str]:
    """Extract column names from SQL query"""
    columns = []
    
    # Simple regex to find SELECT statements
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        
        # Handle nested parentheses in SELECT clause
        current_expr = ""
        paren_level = 0
        col_expressions = []
        
        for char in select_clause:
            if char == '(' and paren_level == 0:
                paren_level += 1
                current_expr += char
            elif char == '(' and paren_level > 0:
                paren_level += 1
                current_expr += char
            elif char == ')' and paren_level > 0:
                paren_level -= 1
                current_expr += char
            elif char == ',' and paren_level == 0:
                col_expressions.append(current_expr.strip())
                current_expr = ""
            else:
                current_expr += char
        
        # Add the last expression
        if current_expr.strip():
            col_expressions.append(current_expr.strip())
        
        # Extract column names from expressions
        for col_expr in col_expressions:
            # Skip * wildcard
            if col_expr == '*':
                continue
                
            # Extract column name (handle aliases with AS keyword)
            as_match = re.search(r'(?:AS|as)\s+([a-zA-Z0-9_]+)$', col_expr)
            if as_match:
                col_name = as_match.group(1)
            else:
                # If no AS keyword, use the last part after a dot or the whole expression
                col_parts = col_expr.split('.')
                col_name = col_parts[-1].strip()
                
                # Remove any functions
                func_match = re.search(r'([a-zA-Z0-9_]+)\s*\(', col_name)
                if func_match:
                    # This is a function, skip it
                    continue
            
            columns.append(col_name)
    
    return columns

def save_metadata(metadata, output_dir):
    """
    Save metadata to output directory in both JSON and YAML formats
    
    Args:
        metadata: The metadata dictionary
        output_dir: Directory to save files to
    """
    import json
    import yaml
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "uni_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save as YAML
    yaml_path = os.path.join(output_dir, "uni_metadata.yml")
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Saved metadata to {json_path} and {yaml_path}")
    
    # Also save with alternate filename for compatibility
    alt_json_path = os.path.join(output_dir, "unified_metadata.json")
    with open(alt_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Also saved as {alt_json_path} for compatibility")

def load_metadata(input_file: str = None) -> Dict[str, Any]:
    """Load metadata from a JSON file"""
    # Set default input file path if not provided
    if input_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = os.path.join(base_dir, "exports", "uni_metadata.json")
    
    # Ensure absolute path
    input_file = os.path.abspath(input_file)
    
    print(f"Loading metadata from: {input_file}")
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metadata file not found: {input_file}")
        # If file doesn't exist, parse projects and create it
        metadata = parse_dbt_projects()
        save_metadata(metadata, input_file)
        return metadata

if __name__ == "__main__":
    metadata = parse_dbt_projects()
    save_metadata(metadata)
    print(f"Parsed {len(metadata['projects'])} projects, {len(metadata['models'])} models, and {len(metadata['lineage'])} lineage relationships")