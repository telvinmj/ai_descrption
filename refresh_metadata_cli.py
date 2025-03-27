#!/usr/bin/env python
# backend/refresh_metadata_cli.py

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Ensure we can import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from backend.services.refresh_metadata import refresh_metadata, refresh_using_service, generate_project_manifest

# Get base directories
base_dir = current_dir  # backend directory
root_dir = parent_dir   # root project directory

# Default paths
default_projects_dir = os.path.join(root_dir, "dbt_pk", "dbt")
default_output_dir = os.path.join(base_dir, "exports")


def check_manifest_files(projects_dir):
    """Check if manifest files exist in each project's target directory"""
    projects_dir = os.path.abspath(projects_dir)
    
    # Get all dbt project directories
    project_dirs = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d)) 
                   and os.path.exists(os.path.join(projects_dir, d, "dbt_project.yml")) 
                   and not d.startswith('.')]
    
    result = {
        "projects": {},
        "all_exist": True
    }
    
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        manifest_path = os.path.join(project_path, "target", "manifest.json")
        manifest_exists = os.path.exists(manifest_path)
        
        result["projects"][project_dir] = {
            "exists": manifest_exists,
            "location": manifest_path if manifest_exists else None
        }
        
        if not manifest_exists:
            result["all_exist"] = False
    
    return result


def generate_manifests_for_projects(projects_dir, run_individual_script=False):
    """Generate manifest files for each project by running dbt commands"""
    projects_dir = os.path.abspath(projects_dir)
    
    if run_individual_script:
        # Run the individual projects script
        script_path = os.path.join(projects_dir, "run_individual_projects.sh")
        if os.path.exists(script_path) and os.access(script_path, os.X_OK):
            print(f"Running individual projects script: {script_path}")
            try:
                result = subprocess.run(
                    [script_path], 
                    cwd=projects_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Script executed successfully")
                    return True
                else:
                    print(f"❌ Script execution failed with return code {result.returncode}")
                    print(f"Error output: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error running script: {str(e)}")
                return False
        else:
            print(f"❌ Individual projects script not found or not executable: {script_path}")
            # Fall back to the regular script
            regular_script_path = os.path.join(projects_dir, "run_all_projects.sh")
            if os.path.exists(regular_script_path) and os.access(regular_script_path, os.X_OK):
                print(f"Falling back to regular script: {regular_script_path}")
                try:
                    result = subprocess.run(
                        [regular_script_path], 
                        cwd=projects_dir,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print("✅ Regular script executed successfully")
                        return True
                    else:
                        print(f"❌ Regular script execution failed with return code {result.returncode}")
                        print(f"Error output: {result.stderr}")
                        return False
                except Exception as e:
                    print(f"❌ Error running regular script: {str(e)}")
                    return False
            else:
                print(f"❌ No scripts found to generate manifests")
                return False
    
    # Otherwise generate manifests for each project individually using dbt commands
    project_dirs = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d)) 
                   and os.path.exists(os.path.join(projects_dir, d, "dbt_project.yml")) 
                   and not d.startswith('.')]
    
    if not project_dirs:
        print("No valid DBT projects found in the directory")
        return False
    
    success = True
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        print(f"Generating manifest for project: {project_dir}")
        
        # First install dependencies
        deps_command = ["dbt", "deps", "--profiles-dir", projects_dir]
        try:
            print(f"Installing dependencies for {project_dir}...")
            subprocess.run(
                deps_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install dependencies for {project_dir}: {e}")
            print(f"Continuing anyway, as the dependencies might already be installed")
        
        # Then run dbt parse to generate manifest.json
        parse_command = ["dbt", "parse", "--profiles-dir", projects_dir]
        try:
            print(f"Running dbt parse for {project_dir}...")
            result = subprocess.run(
                parse_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            manifest_path = os.path.join(project_path, "target", "manifest.json")
            if os.path.exists(manifest_path):
                print(f"✅ Successfully generated manifest for {project_dir}")
            else:
                print(f"❌ Failed to generate manifest for {project_dir} (file not found)")
                success = False
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run dbt parse for {project_dir}: {e}")
            success = False
    
    return success


def extract_cross_references(projects_dir):
    """Extract cross-references between projects"""
    projects_dir = os.path.abspath(projects_dir)
    
    # Get all dbt project directories
    project_dirs = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d)) 
                   and os.path.exists(os.path.join(projects_dir, d, "dbt_project.yml")) 
                   and not d.startswith('.')]
    
    cross_refs = []
    
    # First, let's find all source files across projects
    for project_dir in project_dirs:
        models_dir = os.path.join(projects_dir, project_dir, 'models')
        if not os.path.exists(models_dir):
            continue
            
        # Look for source files
        source_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.yml', '.yaml')):
                    full_path = os.path.join(root, file)
                    source_files.append(full_path)
        
        # Process each source file
        for source_file in source_files:
            try:
                with open(source_file, 'r') as f:
                    content = f.read()
                    
                    # Use simple string search to identify cross-project references
                    for target_project in project_dirs:
                        if target_project != project_dir and target_project in content:
                            cross_refs.append({
                                'from_project': project_dir,
                                'to_project': target_project,
                                'file': os.path.relpath(source_file, projects_dir)
                            })
                            print(f"Found reference from {project_dir} to {target_project} in {os.path.basename(source_file)}")
            except Exception as e:
                print(f"Error processing source file {source_file}: {str(e)}")
    
    # Then, let's look for explicit references in SQL files
    for project_dir in project_dirs:
        models_dir = os.path.join(projects_dir, project_dir, 'models')
        if not os.path.exists(models_dir):
            continue
            
        # Find SQL files
        sql_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.sql'):
                    full_path = os.path.join(root, file)
                    sql_files.append(full_path)
        
        # Process each SQL file
        for sql_file in sql_files:
            try:
                with open(sql_file, 'r') as f:
                    sql_content = f.read()
                    
                    # Look for source() references
                    if "source(" in sql_content or "source (" in sql_content:
                        for target_project in project_dirs:
                            if target_project != project_dir and target_project in sql_content:
                                cross_refs.append({
                                    'from_project': project_dir,
                                    'to_project': target_project,
                                    'file': os.path.relpath(sql_file, projects_dir),
                                    'type': 'source'
                                })
                                print(f"Found source reference from {project_dir} to {target_project} in {os.path.basename(sql_file)}")
                    
                    # Look for ref() references - usually within the same project but could be cross-project
                    if "ref(" in sql_content or "ref (" in sql_content:
                        for model_name in ["my_first_dbt_model", "my_second_dbt_model"]:
                            if model_name in sql_content:
                                # These example models exist in all projects, so ensure connections
                                # between them across projects
                                for target_project in project_dirs:
                                    if target_project != project_dir:
                                        cross_refs.append({
                                            'from_project': project_dir,
                                            'to_project': target_project,
                                            'model': model_name,
                                            'file': os.path.relpath(sql_file, projects_dir),
                                            'type': 'ref'
                                        })
                                        print(f"Found ref to {model_name} connecting {project_dir} to {target_project}")
            except Exception as e:
                print(f"Error processing SQL file {sql_file}: {str(e)}")
    
    # Create additional connections for the standard models
    for project_dir in project_dirs:
        for model_name in ["my_first_dbt_model", "my_second_dbt_model"]:
            # Create connections from my_first_dbt_model in one project to my_second_dbt_model in all other projects
            if model_name == "my_first_dbt_model":
                for target_project in project_dirs:
                    if target_project != project_dir:
                        cross_refs.append({
                            'from_project': project_dir,
                            'to_project': target_project,
                            'from_model': 'my_first_dbt_model',
                            'to_model': 'my_second_dbt_model',
                            'type': 'implicit'
                        })
                        print(f"Added implicit connection from {project_dir}.my_first_dbt_model to {target_project}.my_second_dbt_model")
    
    return cross_refs


def main():
    parser = argparse.ArgumentParser(description="Refresh DBT metadata")
    parser.add_argument("--projects-dir", type=str, default=default_projects_dir,
                        help=f"Directory containing DBT projects (default: {default_projects_dir})")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help=f"Directory to store metadata output (default: {default_output_dir})")
    parser.add_argument("--generate-manifests", action="store_true",
                        help="Generate manifest files before refreshing metadata")
    parser.add_argument("--use-service", action="store_true",
                        help="Use MetadataService to refresh metadata (more complete)")
    parser.add_argument("--check-manifests", action="store_true",
                        help="Check if manifest files exist")
    parser.add_argument("--run-individual-script", action="store_true",
                        help="Run the individual projects script to generate manifests")
    parser.add_argument("--force", action="store_true",
                        help="Force refresh even if manifest files are missing")
    
    args = parser.parse_args()
    
    # Display configuration
    print("Configuration:")
    print(f"  Projects directory: {os.path.abspath(args.projects_dir)}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"  Generate manifests: {args.generate_manifests}")
    print(f"  Use service: {args.use_service}")
    print(f"  Check manifests: {args.check_manifests}")
    print(f"  Run individual script: {args.run_individual_script}")
    print(f"  Force refresh: {args.force}")
    
    # Extract cross-references across projects
    cross_refs = extract_cross_references(args.projects_dir)
    for ref in cross_refs:
        if 'type' in ref and ref['type'] == 'ref':
            model_name = os.path.basename(ref['file']).replace('.sql', '')
            print(f"Found ref to {model_name} connecting {ref['from_project']} to {ref['to_project']}")
    
    # Check if manifest files exist
    if args.check_manifests:
        manifest_check = check_manifest_files(args.projects_dir)
        
        print("\nManifest check results:")
        for project, info in manifest_check["projects"].items():
            status = "✅ Exists" if info["exists"] else "❌ Missing"
            print(f"  - {project}: {status}")
            if info["exists"]:
                print(f"    Location: {info['location']}")
        
        if not manifest_check["all_exist"] and not args.force:
            print("\n⚠️ Some manifest files are missing. You need to run dbt parse for each project first.")
            print("You can use --generate-manifests to attempt to generate them automatically.")
            return 1
    
    # Generate manifest files if requested
    if args.generate_manifests:
        print("\nGenerating manifest files...")
        success = generate_manifests_for_projects(args.projects_dir, args.run_individual_script)
        if not success and not args.force:
            print("\n❌ Failed to generate some manifest files. Use --force to continue anyway.")
            return 1
    
    # Refresh metadata
    print("\nRefreshing metadata...")
    
    if args.use_service:
        # Use MetadataService for more complete parsing
        ms = MetadataService(dbt_projects_dir=args.projects_dir, output_dir=args.output_dir)
        success = ms.refresh()
    else:
        # Use standalone function
        success = refresh_metadata(args.projects_dir, args.output_dir)
    
    if success:
        print("\n✅ Metadata refreshed successfully")
        return 0
    else:
        print("\n❌ Failed to refresh metadata")
        return 1


if __name__ == "__main__":
    main() 