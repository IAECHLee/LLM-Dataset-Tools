import os
import shutil
import re
from pathlib import Path

def restore_images_to_original_folders(source_folder):
    """
    Restore images from Pattern1_right and Pattern1_left back to their original pattern folders.
    Uses the timestamp in filename to determine the original folder.
    
    Args:
        source_folder: Path to the folder containing Pattern*_right and Pattern*_left folders
    """
    source_path = Path(source_folder)
    
    if not source_path.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    print(f"Source folder: {source_path}")
    print("-" * 60)
    
    # Find all organized folders (Pattern1_right, Pattern1_left, etc.)
    organized_folders = []
    for folder in source_path.iterdir():
        if folder.is_dir() and ('_right' in folder.name.lower() or '_left' in folder.name.lower()):
            organized_folders.append(folder)
    
    if not organized_folders:
        print("No organized folders (Pattern*_right or Pattern*_left) found.")
        return
    
    print(f"Found {len(organized_folders)} organized folders to restore from")
    print()
    
    # Statistics
    total_moved = 0
    total_errors = 0
    
    for org_folder in organized_folders:
        print(f"Processing: {org_folder.name}")
        moved_count = 0
        error_count = 0
        
        # Process all files in this organized folder
        for file_path in org_folder.iterdir():
            if not file_path.is_file():
                continue
            
            filename = file_path.name
            
            # Extract timestamp from filename (e.g., "2025-11-17 18-06-36.343")
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})', filename)
            if not timestamp_match:
                print(f"  ⚠ Cannot parse timestamp: {filename}")
                error_count += 1
                continue
            
            timestamp = timestamp_match.group(1)
            
            # Find matching original folder
            found_folder = None
            for subfolder in source_path.iterdir():
                if not subfolder.is_dir():
                    continue
                if '_right' in subfolder.name.lower() or '_left' in subfolder.name.lower():
                    continue
                if timestamp in subfolder.name:
                    found_folder = subfolder
                    break
            
            if found_folder:
                # Move file back to original folder
                dest_path = found_folder / filename
                
                # Handle duplicates
                counter = 1
                while dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    dest_path = found_folder / f"{stem}_restored_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(file_path), str(dest_path))
                moved_count += 1
            else:
                print(f"  ⚠ No matching folder for: {filename}")
                error_count += 1
        
        print(f"  Restored: {moved_count}")
        if error_count > 0:
            print(f"  Errors: {error_count}")
        
        total_moved += moved_count
        total_errors += error_count
        
        # Remove empty folder
        if not any(org_folder.iterdir()):
            org_folder.rmdir()
            print(f"  Removed empty folder: {org_folder.name}")
    
    print("-" * 60)
    print(f"\nRestore complete!")
    print(f"  Total restored: {total_moved}")
    if total_errors > 0:
        print(f"  Total errors: {total_errors}")


if __name__ == "__main__":
    # Source folder path
    source_folder = r"F:\Image storage"
    
    print("=" * 60)
    print("Image Restore - Return to Original Folders")
    print("=" * 60)
    print()
    
    # Check if source exists
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' not found.")
        print("Please check the path and try again.")
    else:
        print(f"This will restore images from organized folders back to:")
        print(f"  {source_folder}\\[original timestamp folders]")
        print()
        
        response = input("Continue? (y/n): ").strip().lower()
        if response == 'y':
            restore_images_to_original_folders(source_folder)
        else:
            print("Cancelled.")
