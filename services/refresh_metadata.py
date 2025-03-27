#!/usr/bin/env python
# backend/services/refresh_metadata.py

import os
import sys
import json
import subprocess
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

# Ensure we can import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from backend.services.dbt_metadata_parser import parse_dbt_projects, save_metadata
from backend.services.metadata_service import MetadataService


def run_dbt_command(command, project_dir=None, profiles_dir=None):
    """Run a dbt command and return the output"""
    cmd = ["dbt"] + command
    
    env = os.environ.copy()
    if project_dir:
        env["DBT_PROJECT_DIR"] = project_dir
    
    if profiles_dir:
        cmd.extend(["--profiles-dir", profiles_dir])
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=project_dir or os.getcwd(),
            env=env,
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running dbt command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return None


def generate_manifests(projects_dir):
    """Generate manifest.json files for each dbt project"""
    
    # Ensure the main profiles.yml is accessible
    profiles_dir = os.path.abspath(projects_dir)
    
    # Get all project directories
    project_dirs = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d)) 
                   and not d.startswith('.')]
    
    # Remove directories that don't look like dbt projects
    valid_project_dirs = []
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        project_file = os.path.join(project_path, "dbt_project.yml")
        if os.path.exists(project_file):
            valid_project_dirs.append(project_dir)
    
    print(f"Found {len(valid_project_dirs)} valid dbt projects: {', '.join(valid_project_dirs)}")
    
    # Try running the run_individual_projects.sh script if it exists
    individual_script = os.path.join(projects_dir, "run_individual_projects.sh")
    if os.path.exists(individual_script) and os.access(individual_script, os.X_OK):
        print(f"Found run_individual_projects.sh script, running it to generate manifests...")
        try:
            result = subprocess.run(
                [individual_script],
                cwd=projects_dir,
                capture_output=True,
                text=True
            )
            print(f"Run script completed with exit code {result.returncode}")
            if result.returncode != 0:
                print(f"Script output: {result.stdout}")
                print(f"Script error: {result.stderr}")
            else:
                return True
        except Exception as e:
            print(f"Error running script: {str(e)}")
    # Fall back to run_all_projects.sh if individual script doesn't exist
    elif os.path.exists(os.path.join(projects_dir, "run_all_projects.sh")) and os.access(os.path.join(projects_dir, "run_all_projects.sh"), os.X_OK):
        run_all_script = os.path.join(projects_dir, "run_all_projects.sh")
        print(f"Found run_all_projects.sh script, running it to generate manifests...")
        try:
            result = subprocess.run(
                [run_all_script],
                cwd=projects_dir,
                capture_output=True,
                text=True
            )
            print(f"Run script completed with exit code {result.returncode}")
            if result.returncode != 0:
                print(f"Script output: {result.stdout}")
                print(f"Script error: {result.stderr}")
            else:
                return True
        except Exception as e:
            print(f"Error running script: {str(e)}")
    else:
        # Generate manifests for each project
        for project_dir in valid_project_dirs:
            project_path = os.path.join(projects_dir, project_dir)
            print(f"Generating manifest for {project_dir}...")
            
            # Run dbt deps first
            deps_result = run_dbt_command(
                ["deps"], 
                project_dir=project_path,
                profiles_dir=profiles_dir
            )
            
            if not deps_result:
                print(f"Warning: Failed to install dependencies for {project_dir}, continuing anyway...")
            
            # Run dbt parse to generate manifest
            result = run_dbt_command(
                ["parse"], 
                project_dir=project_path,
                profiles_dir=profiles_dir
            )
            
            if result:
                print(f"Successfully generated manifest for {project_dir}")
            else:
                print(f"Failed to generate manifest for {project_dir}")
    
    # Check if manifests were created for each project
    success = True
    for project_dir in valid_project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        manifest_path = os.path.join(project_path, "target", "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"Warning: No manifest file found for {project_dir}")
            success = False
    
    return success


def find_manifest_files(projects_dir):
    """Find all manifest.json files in the projects directory"""
    manifest_files = []
    
    # Get all project directories
    project_dirs = [d for d in os.listdir(projects_dir) 
                   if os.path.isdir(os.path.join(projects_dir, d)) 
                   and os.path.exists(os.path.join(projects_dir, d, "dbt_project.yml")) 
                   and not d.startswith('.')]
    
    # First, check for manifest files in each project's target directory
    for project_dir in project_dirs:
        project_path = os.path.join(projects_dir, project_dir)
        target_dir = os.path.join(project_path, "target")
        
        if os.path.isdir(target_dir):
            manifest_path = os.path.join(target_dir, "manifest.json")
            if os.path.exists(manifest_path):
                manifest_files.append(manifest_path)
                print(f"Found manifest for project '{project_dir}' at {manifest_path}")
    
    # If no manifest files found in expected locations, do a more comprehensive search
    if not manifest_files:
        print("No manifest files found in expected locations, performing deep search...")
        for root, dirs, files in os.walk(projects_dir):
            if "manifest.json" in files:
                manifest_path = os.path.join(root, "manifest.json")
                manifest_files.append(manifest_path)
                print(f"Found manifest file during deep search: {manifest_path}")
    
    return manifest_files


def generate_project_manifest(project_path, profiles_dir=None):
    """Generate manifest.json file for a specific dbt project"""
    project_path = os.path.abspath(project_path)
    project_name = os.path.basename(project_path)
    
    print(f"Generating manifest for project {project_name} at {project_path}")
    
    # Set profiles_dir to the parent directory by default if not specified
    if profiles_dir is None:
        profiles_dir = os.path.dirname(project_path)
    
    # Check if the project has a dbt_project.yml file
    project_file = os.path.join(project_path, "dbt_project.yml")
    if not os.path.exists(project_file):
        print(f"Error: {project_name} is not a valid dbt project (no dbt_project.yml found)")
        return False
    
    # Run dbt deps first
    try:
        deps_result = run_dbt_command(
            ["deps"], 
            project_dir=project_path,
            profiles_dir=profiles_dir
        )
        
        if not deps_result:
            print(f"Warning: Failed to install dependencies for {project_name}, continuing anyway...")
    except Exception as e:
        print(f"Error installing dependencies for {project_name}: {str(e)}")
        # Continue anyway, as dependencies might be installed already
    
    # Run dbt parse to generate manifest
    try:
        result = run_dbt_command(
            ["parse"], 
            project_dir=project_path,
            profiles_dir=profiles_dir
        )
        
        # Check if target/manifest.json was created
        manifest_path = os.path.join(project_path, "target", "manifest.json")
        if result and os.path.exists(manifest_path):
            print(f"Successfully generated manifest at {manifest_path}")
            return True
        else:
            print(f"Failed to generate manifest for {project_name}")
            return False
            
    except Exception as e:
        print(f"Error generating manifest for {project_name}: {str(e)}")
        return False


def refresh_metadata(projects_dir: str = "dbt_projects_2", output_dir: str = None) -> bool:
    """
    Parse dbt projects and refresh the unified metadata
    
    Args:
        projects_dir: Directory containing dbt projects
        output_dir: Directory to store the exported metadata
        
    Returns:
        bool: True if refresh was successful, False otherwise
    """
    try:
        print("\n=== Starting metadata refresh ===")
        
        # Get the base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(base_dir, "exports")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get absolute paths
        if not os.path.isabs(projects_dir):
            # If projects_dir is relative, make it absolute relative to parent of base_dir
            projects_dir = os.path.abspath(os.path.join(os.path.dirname(base_dir), projects_dir))
        else:
            projects_dir = os.path.abspath(projects_dir)
            
        output_dir = os.path.abspath(output_dir)
        output_file = os.path.join(output_dir, "uni_metadata.json")
        
        print(f"Base directory: {base_dir}")
        print(f"Projects directory: {projects_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Output file: {output_file}")
        
        # Check if projects directory exists
        if not os.path.exists(projects_dir):
            print(f"Error: Projects directory not found: {projects_dir}")
            return False
        
        # Get list of project directories
        project_dirs = [d for d in os.listdir(projects_dir) 
                       if os.path.isdir(os.path.join(projects_dir, d)) 
                       and os.path.exists(os.path.join(projects_dir, d, "dbt_project.yml"))
                       and not d.startswith('.')]
        
        if not project_dirs:
            print(f"Warning: No dbt project directories found in {projects_dir}")
            # Create empty metadata structure if no projects found
            metadata = {
                "projects": [],
                "models": [],
                "lineage": []
            }
        else:
            print(f"Found {len(project_dirs)} dbt project directories: {', '.join(project_dirs)}")
            
            # Check if each project has a manifest file
            missing_manifests = []
            for project_dir in project_dirs:
                project_path = os.path.join(projects_dir, project_dir)
                manifest_path = os.path.join(project_path, "target", "manifest.json")
                if not os.path.exists(manifest_path):
                    missing_manifests.append(project_dir)
            
            if missing_manifests:
                print(f"Warning: The following projects are missing manifest files: {', '.join(missing_manifests)}")
                print("Attempting to generate missing manifest files...")
                
                for project_dir in missing_manifests:
                    project_path = os.path.join(projects_dir, project_dir)
                    if not generate_project_manifest(project_path, profiles_dir=projects_dir):
                        print(f"Failed to generate manifest for {project_dir}")
            
            # Now use the existing parsing function
            metadata = parse_dbt_projects(projects_dir)
            
            if metadata and 'projects' in metadata and len(metadata['projects']) == 0:
                print("Warning: No projects were parsed from the manifest files.")
                print("This might indicate an issue with the manifest files or the parsing logic.")
        
        # Save the metadata
        save_metadata(metadata, output_dir)
        
        print(f"\nMetadata refresh completed successfully:")
        print(f"- Projects: {len(metadata.get('projects', []))}")
        print(f"- Models: {len(metadata.get('models', []))}")
        print(f"- Lineage: {len(metadata.get('lineage', []))}")
        
        return True
        
    except Exception as e:
        print(f"Error refreshing metadata: {str(e)}")
        return False


def refresh_using_service(projects_dir=None, output_dir=None, cross_refs=None):
    """
    Refresh metadata using the MetadataService which handles all the processing.
    This is the preferred method as it uses the full service capabilities.
    """
    try:
        from backend.services.metadata_service import MetadataService
        
        # Configure the service with appropriate directories
        service = MetadataService(
            dbt_projects_dir=projects_dir,
            output_dir=output_dir
        )
        
        # If we have cross-references, add them to the service
        if cross_refs:
            print(f"Adding {len(cross_refs)} cross-project references to enhance lineage")
            service.add_cross_references(cross_refs)
        
        # Perform refresh
        result = service.refresh()
        
        if result:
            print(f"✅ Metadata refresh completed successfully using service")
            print(f"Output saved to: {service.unified_metadata_path}")
            return True
        else:
            print(f"❌ Metadata refresh failed")
            return False
            
    except Exception as e:
        print(f"❌ Error using MetadataService: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_metadata(metadata, output_dir):
    """Save metadata to the output directory"""
    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full unified metadata
    output_file = os.path.join(output_dir, "uni_metadata.json")
    with open(output_file, 'w') as f:  # Use 'w' mode to overwrite existing file
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {output_file}")
    
    # Save individual metadata files for frontend
    projects_file = os.path.join(output_dir, "projects.json")
    models_file = os.path.join(output_dir, "models.json")
    lineage_file = os.path.join(output_dir, "lineage.json")
    
    with open(projects_file, 'w') as f:
        json.dump(metadata.get("projects", []), f, indent=2)
    print(f"Saved projects to {projects_file}")
    
    with open(models_file, 'w') as f:
        json.dump(metadata.get("models", []), f, indent=2)
    print(f"Saved models to {models_file}")
    
    with open(lineage_file, 'w') as f:
        json.dump(metadata.get("lineage", []), f, indent=2)
    print(f"Saved lineage to {lineage_file}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh DBT project metadata")
    parser.add_argument("--projects-dir", default="dbt_projects_2", help="Directory containing dbt projects")
    parser.add_argument("--output-dir", default="backend/exports", help="Directory to store the exported metadata")
    parser.add_argument("--use-service", action="store_true", help="Use MetadataService instead of direct functions")
    
    args = parser.parse_args()
    
    if args.use_service:
        success = refresh_using_service()
    else:
        success = refresh_metadata(args.projects_dir, args.output_dir)
    
    if success:
        print("\n✅ Metadata refresh completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Metadata refresh failed")
        sys.exit(1) 