import os
import shutil
from pathlib import Path
from datetime import datetime
import re

def parse_filename_time(filename):
    """Extract datetime from filename"""
    pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2})-(\d{2})-(\d{2})\.(\d{3})'
    match = re.match(pattern, filename)
    
    if match:
        date_str = match.group(1)
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)
        millisecond = match.group(5)
        
        datetime_str = f"{date_str} {hour}:{minute}:{second}.{millisecond}"
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return None
    return None

def organize_by_major_patterns(source_folder, gap_threshold_minutes=50):
    """
    Organize files into major patterns based on large time gaps
    
    Args:
        source_folder: Base folder containing Pattern1_left and Pattern1_right
        gap_threshold_minutes: Gap size in minutes to consider as pattern break (default 50)
    """
    source_path = Path(source_folder)
    
    print("=" * 80)
    print("ORGANIZING BY MAJOR PATTERNS")
    print("=" * 80)
    print(f"Source folder: {source_path}")
    print(f"Gap threshold: {gap_threshold_minutes} minutes")
    print()
    
    # Define the major gap times based on analysis
    # These are the timestamps where gaps >= 50 minutes occur
    major_gaps = [
        datetime(2025, 11, 17, 20, 32, 13),  # After Session 6 (63.4 min)
        datetime(2025, 11, 17, 22, 26, 23),  # After Session 9 (59.3 min)
        datetime(2025, 11, 18, 0, 47, 40),   # After Session 13 (50.2 min)
        datetime(2025, 11, 18, 1, 59, 21),   # After Session 14 (57.2 min)
        datetime(2025, 11, 18, 4, 33, 24),   # After Session 19 (108.8 min)
        datetime(2025, 11, 18, 6, 26, 18),   # After Session 20 (53.9 min)
        datetime(2025, 11, 18, 9, 9, 3),     # After Session 24 (57.9 min)
        datetime(2025, 11, 18, 10, 54, 30),  # After Session 26 (58.9 min)
        datetime(2025, 11, 18, 11, 57, 23),  # After Session 27 (84.5 min)
    ]
    
    # Process both left and right folders
    for direction in ['left', 'right']:
        folder_name = f"Pattern1_{direction}"
        folder_path = source_path / folder_name
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {folder_name}...")
        print("-" * 80)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(files)} files")
        
        # Parse times and sort
        file_times = []
        for file in files:
            dt = parse_filename_time(file.name)
            if dt:
                file_times.append((dt, file))
        
        file_times.sort(key=lambda x: x[0])
        print(f"Valid timestamps: {len(file_times)}")
        
        # Determine pattern number for each file
        pattern_files = {}  # pattern_num -> list of files
        
        for dt, file_path in file_times:
            # Find which pattern this file belongs to
            pattern_num = 1
            for gap_time in major_gaps:
                if dt > gap_time:
                    pattern_num += 1
                else:
                    break
            
            if pattern_num not in pattern_files:
                pattern_files[pattern_num] = []
            pattern_files[pattern_num].append(file_path)
        
        print(f"\nFound {len(pattern_files)} major patterns")
        for pattern_num in sorted(pattern_files.keys()):
            print(f"  Pattern {pattern_num}: {len(pattern_files[pattern_num])} files")
        
        # Create pattern folders and move files
        print(f"\nCreating pattern folders and moving files...")
        
        for pattern_num, files_list in pattern_files.items():
            # Create pattern folder: Pattern1/left or Pattern1/right
            pattern_folder = source_path / f"Pattern{pattern_num}" / direction
            pattern_folder.mkdir(parents=True, exist_ok=True)
            
            # Move files
            moved = 0
            for file_path in files_list:
                dest_path = pattern_folder / file_path.name
                
                # Handle duplicates
                counter = 1
                while dest_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    dest_path = pattern_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.move(str(file_path), str(dest_path))
                    moved += 1
                except Exception as e:
                    print(f"  Error moving {file_path.name}: {e}")
            
            print(f"  Pattern {pattern_num}/{direction}: Moved {moved}/{len(files_list)} files")
        
        print(f"\nCompleted {folder_name}")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print()
    print("Created pattern structure:")
    for pattern_num in range(1, len(major_gaps) + 2):
        print(f"  Pattern{pattern_num}/")
        print(f"    ├── left/")
        print(f"    └── right/")
    
    # Cleanup: Remove empty Pattern1_left and Pattern1_right folders
    print("\nCleaning up empty folders...")
    for direction in ['left', 'right']:
        folder_name = f"Pattern1_{direction}"
        folder_path = source_path / folder_name
        if folder_path.exists() and not any(folder_path.iterdir()):
            folder_path.rmdir()
            print(f"  Removed empty folder: {folder_name}")

if __name__ == "__main__":
    source_folder = r"F:\Image storage"
    
    print("=" * 80)
    print("PATTERN ORGANIZER - Major Pattern Separation")
    print("=" * 80)
    print()
    print("This will organize files into 9 major patterns based on large time gaps.")
    print()
    
    organize_by_major_patterns(source_folder, gap_threshold_minutes=50)
