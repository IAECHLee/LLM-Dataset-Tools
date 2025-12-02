import os
import shutil
import re
from pathlib import Path

def organize_images_by_pattern_and_direction(source_folder):
    """
    Organize images into Pattern-specific folders (Pattern1_right, Pattern1_left, Pattern2_right, etc.)
    based on the source subfolder name.
    
    Args:
        source_folder: Path to the folder containing pattern subfolders
    """
    source_path = Path(source_folder)
    
    if not source_path.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    print(f"Source folder: {source_path}")
    print("-" * 60)
    
    # Image extensions to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Statistics per pattern
    pattern_stats = {}
    
    # Find all pattern folders (e.g., folders containing "Pattern1", "Pattern2", etc.)
    for subfolder in source_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Skip already organized folders
        if '_right' in subfolder.name.lower() or '_left' in subfolder.name.lower():
            continue
        
        # Extract pattern number from folder name (e.g., "2025-11-17 18-06-36(Pattern1)" -> "Pattern1")
        pattern_match = re.search(r'\(Pattern(\d+)\)', subfolder.name, re.IGNORECASE)
        if not pattern_match:
            continue
        
        pattern_num = pattern_match.group(1)
        pattern_name = f"Pattern{pattern_num}"
        
        print(f"\nProcessing: {subfolder.name} -> {pattern_name}")
        
        # Create output folders for this pattern
        right_folder = source_path / f"{pattern_name}_right"
        left_folder = source_path / f"{pattern_name}_left"
        
        right_folder.mkdir(exist_ok=True)
        left_folder.mkdir(exist_ok=True)
        
        right_count = 0
        left_count = 0
        skipped_count = 0
        
        # Process all files in this subfolder
        for file_path in subfolder.rglob('*'):
            if not file_path.is_file():
                continue
                
            # Check if it's an image file
            if file_path.suffix.lower() not in image_extensions:
                continue
            
            filename = file_path.name
            filename_lower = filename.lower()
            
            # Determine destination based on filename content (using fullwidth semicolon ；)
            if '；right；' in filename_lower:
                # Move to right folder
                dest_path = right_folder / filename
                # Handle duplicate filenames
                counter = 1
                while dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    dest_path = right_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(file_path), str(dest_path))
                right_count += 1
                
            elif '；left；' in filename_lower:
                # Move to left folder
                dest_path = left_folder / filename
                # Handle duplicate filenames
                counter = 1
                while dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    dest_path = left_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(file_path), str(dest_path))
                left_count += 1
                
            else:
                # No direction found in filename
                skipped_count += 1
        
        # Store statistics
        pattern_stats[pattern_name] = {
            'right': right_count,
            'left': left_count,
            'skipped': skipped_count
        }
        
        print(f"  → Right: {right_count}")
        print(f"  ← Left: {left_count}")
        print(f"  ⊘ Skipped: {skipped_count}")
    
    print("-" * 60)
    print(f"\nOrganization complete!")
    print("\nSummary by Pattern:")
    total_right = 0
    total_left = 0
    total_skipped = 0
    
    for pattern_name in sorted(pattern_stats.keys()):
        stats = pattern_stats[pattern_name]
        print(f"\n{pattern_name}:")
        print(f"  Right: {stats['right']}")
        print(f"  Left: {stats['left']}")
        print(f"  Skipped: {stats['skipped']}")
        total_right += stats['right']
        total_left += stats['left']
        total_skipped += stats['skipped']
    
    print(f"\nGrand Total:")
    print(f"  Right images: {total_right}")
    print(f"  Left images: {total_left}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Total processed: {total_right + total_left + total_skipped}")


if __name__ == "__main__":
    # Source folder path
    source_folder = r"F:\Image storage"
    
    print("=" * 60)
    print("Image Organizer - Right/Left Separation")
    print("=" * 60)
    print()
    
    # Check if source exists
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' not found.")
        print("Please check the path and try again.")
    else:
        # Ask for confirmation
        print(f"This will organize images from:")
        print(f"  {source_folder}")
        print()
        print(f"Images will be moved to:")
        print(f"  {source_folder}\\Pattern1_right")
        print(f"  {source_folder}\\Pattern1_left")
        print()
        
        # Auto-run without confirmation prompt
        organize_images_by_pattern_and_direction(source_folder)
