import os
import sys
import uvicorn
import time
import argparse
from dotenv import load_dotenv

# Add the parent directory to sys.path so Python can find the backend module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ==============================
# EDIT THIS VALUE TO CHANGE THE DBT PROJECTS DIRECTORY
# This can be a relative path (from the backend directory) or an absolute path
# Examples:
#   - "../dbt_projects_2"
#   - "../my_custom_projects"  
#   - "/Users/telvin/Desktop/dbt/some_other_projects"
# ==============================
DEFAULT_PROJECTS_DIR = "/Users/telvin/Desktop/dbt/pk"  # <-- EDIT THIS LINE
# ==============================

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="DBT Metadata Explorer Backend Server")
    parser.add_argument(
        "--projects-dir", 
        dest="projects_dir",
        default=DEFAULT_PROJECTS_DIR,  # Use the hardcoded value as default
        help=f"Directory containing dbt projects (default: {DEFAULT_PROJECTS_DIR})"
    )
    parser.add_argument(
        "--host", 
        default=None,
        help="Host to run the server on (default: uses API_HOST env var or 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=None,
        help="Port to run the server on (default: uses API_PORT env var or 8000)"
    )
    return parser.parse_args()

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    args = parse_args()
    
    # Store the projects_dir in the environment so it can be accessed by other modules
    os.environ["DBT_PROJECTS_DIR"] = args.projects_dir
    
    # Check for Gemini API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("\n⚠️  WARNING: GEMINI_API_KEY environment variable not found!")
        print("AI description generation will be disabled.")
        print("To enable AI descriptions, set your Gemini API key:")
        print("  export GEMINI_API_KEY='your-api-key'\n")
        print("You can get a Gemini API key from: https://makersuite.google.com/app/apikey\n")
        time.sleep(2)  # Give user time to read the message
    
    # Run the app
    host = args.host or os.getenv("API_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("API_PORT", "8000"))
    
    print(f"Starting backend server at http://{host}:{port}")
    print(f"Using dbt projects directory: {args.projects_dir}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
