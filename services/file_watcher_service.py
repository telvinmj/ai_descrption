import os
import time
import threading
from typing import Callable, List, Dict, Any, Set, Optional
from datetime import datetime
import json

class FileWatcherService:
    """Service to watch for changes in dbt project files and trigger metadata refresh"""
    
    def __init__(self, 
                 dbt_projects_dir: str, 
                 refresh_callback: Callable[[], bool],
                 watch_interval: int = 60,  # Check every 60 seconds
                 file_patterns: List[str] = None):
        """
        Initialize the file watcher service
        
        Args:
            dbt_projects_dir: Directory containing dbt projects
            refresh_callback: Function to call when changes are detected
            watch_interval: How often to check for changes in seconds
            file_patterns: List of file patterns to watch for (default: manifest.json, catalog.json)
        """
        # If already absolute, use it directly; otherwise, make it absolute
        self.dbt_projects_dir = dbt_projects_dir if os.path.isabs(dbt_projects_dir) else os.path.abspath(dbt_projects_dir)
        self.refresh_callback = refresh_callback
        self.watch_interval = watch_interval
        self.file_patterns = file_patterns or ["manifest.json", "catalog.json"]
        self.last_refresh_time = datetime.now()
        self.watcher_thread = None
        self.watching = False
        self.file_timestamps: Dict[str, float] = {}
        
        # Initial scan to store file timestamps
        self._scan_files()
    
    def _scan_files(self) -> Dict[str, float]:
        """Scan all projects and store file timestamps"""
        new_timestamps = {}
        
        # Start by checking if directory exists
        if not os.path.exists(self.dbt_projects_dir):
            print(f"Warning: dbt projects directory {self.dbt_projects_dir} does not exist")
            return new_timestamps
        
        for root, dirs, files in os.walk(self.dbt_projects_dir):
            for file in files:
                if any(pattern in file for pattern in self.file_patterns):
                    file_path = os.path.join(root, file)
                    try:
                        mod_time = os.path.getmtime(file_path)
                        new_timestamps[file_path] = mod_time
                    except Exception as e:
                        print(f"Error accessing file {file_path}: {str(e)}")
        
        return new_timestamps
    
    def _check_for_changes(self) -> bool:
        """Check if any watched files have changed since last refresh"""
        new_timestamps = self._scan_files()
        
        # No files found
        if not new_timestamps:
            return False
        
        # First run, store timestamps
        if not self.file_timestamps:
            self.file_timestamps = new_timestamps
            return False
        
        # Check for new or modified files
        for file_path, mod_time in new_timestamps.items():
            if file_path not in self.file_timestamps or mod_time > self.file_timestamps[file_path]:
                print(f"Detected change in file: {file_path}")
                self.file_timestamps = new_timestamps
                return True
        
        # Check for deleted files
        for file_path in self.file_timestamps.keys():
            if file_path not in new_timestamps:
                print(f"Detected deleted file: {file_path}")
                self.file_timestamps = new_timestamps
                return True
        
        return False
    
    def _watcher_loop(self):
        """Main watcher loop that runs in a separate thread"""
        print(f"Starting file watcher for dbt projects in {self.dbt_projects_dir}")
        print(f"Watching for changes in: {', '.join(self.file_patterns)}")
        
        while self.watching:
            try:
                if self._check_for_changes():
                    print("Changes detected, refreshing metadata...")
                    success = self.refresh_callback()
                    if success:
                        self.last_refresh_time = datetime.now()
                        print(f"Metadata refreshed successfully at {self.last_refresh_time}")
                    else:
                        print("Metadata refresh failed")
            except Exception as e:
                print(f"Error in file watcher: {str(e)}")
            
            # Sleep for the watch interval
            time.sleep(self.watch_interval)
    
    def start(self):
        """Start the file watcher in a separate thread"""
        if not self.watching:
            self.watching = True
            self.watcher_thread = threading.Thread(target=self._watcher_loop, daemon=True)
            self.watcher_thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the file watcher"""
        if self.watching:
            self.watching = False
            if self.watcher_thread and self.watcher_thread.is_alive():
                self.watcher_thread.join(timeout=2)
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the watcher"""
        return {
            "active": self.watching,
            "watching_directory": self.dbt_projects_dir,
            "watching_patterns": self.file_patterns,
            "watch_interval_seconds": self.watch_interval,
            "watched_files_count": len(self.file_timestamps),
            "last_refresh_time": self.last_refresh_time.isoformat() if self.last_refresh_time else None
        } 