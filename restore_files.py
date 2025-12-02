import os
import shutil
from pathlib import Path
import re

def restore_files_to_original_folders(source_folder):
    """
    Restore files from Pattern1_right and Pattern1_left back to their original dated folders.
    Uses timestamp from filename to match the original folder.
    """
    source_path = Path(source_folder)
    
    # Pattern folders to restore from
    pattern_folders = [
        source_path / "Pattern1_right",
        source_path / "Pattern1_left"
    ]
    
    restored_count = 0
    error_count = 0
    
    print("=" * 60)
    print("Restoring files to original folders...")
    print("=" * 60)
    
    for pattern_folder in pattern_folders:
        if not pattern_folder.exists():
            print(f"Folder not found: {pattern_folder}")
            continue
        
        print(f"\nProcessing: {pattern_folder.name}")
        print("-" * 60)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = [f for f in pattern_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        for file_path in files:
            filename = file_path.name
            
            # Extract timestamp from filename (format: YYYY-MM-DD HH-MM-SS.mmm)
            # Example: 2025-11-17 18-06-36.343；right；...
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})', filename)
            
            if not timestamp_match:
                print(f"⚠ No timestamp found: {filename}")
                error_count += 1
                continue
            
            timestamp = timestamp_match.group(1)
            
            # Find matching folder in source directory
            # Folder format: "2025-11-17 18-06-36(Pattern1)" or similar
            matching_folders = []
            for item in source_path.iterdir():
                if item.is_dir() and timestamp in item.name:
                    matching_folders.append(item)
            
            if not matching_folders:
                print(f"⚠ No matching folder for: {filename}")
                error_count += 1
                continue
            
            if len(matching_folders) > 1:
                print(f"⚠ Multiple folders match {filename}: {[f.name for f in matching_folders]}")
                error_count += 1
                continue
            
            # Move file back to original folder
            dest_folder = matching_folders[0]
            dest_path = dest_folder / filename
            
            # Handle duplicate filenames
            counter = 1
            while dest_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                dest_path = dest_folder / f"{stem}_restored_{counter}{suffix}"
                counter += 1
            
            try:
                shutil.move(str(file_path), str(dest_path))
                restored_count += 1
                if restored_count % 100 == 0:
                    print(f"✓ Restored {restored_count} files...")
            except Exception as e:
                print(f"✗ Error moving {filename}: {e}")
                error_count += 1
    
    print("\n" + "=" * 60)
    print("Restore complete!")
    print(f"  Files restored: {restored_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    source_folder = r"F:\Image storage"
    
    print("\nThis will restore files from Pattern1_right and Pattern1_left")
    print("back to their original dated folders.\n")
    
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response == 'yes':
        restore_files_to_original_folders(source_folder)
    else:
        print("Cancelled.")
