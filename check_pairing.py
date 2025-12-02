import os
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

def check_pairing(left_folder, right_folder):
    """Check if left and right files are properly paired"""
    
    left_path = Path(left_folder)
    right_path = Path(right_folder)
    
    print("=" * 80)
    print("CHECKING LEFT-RIGHT PAIRING")
    print("=" * 80)
    print()
    
    # Get all files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    print("Reading left files...")
    left_files = [f for f in left_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"Found {len(left_files)} left files")
    
    print("Reading right files...")
    right_files = [f for f in right_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"Found {len(right_files)} right files")
    
    print()
    print("Parsing timestamps...")
    
    # Create timestamp sets
    left_times = {}
    for f in left_files:
        dt = parse_filename_time(f.name)
        if dt:
            # Use timestamp as key (rounded to millisecond)
            key = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            left_times[key] = f.name
    
    right_times = {}
    for f in right_files:
        dt = parse_filename_time(f.name)
        if dt:
            key = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            right_times[key] = f.name
    
    print(f"Left timestamps parsed: {len(left_times)}")
    print(f"Right timestamps parsed: {len(right_times)}")
    print()
    
    # Find unpaired files
    left_only = set(left_times.keys()) - set(right_times.keys())
    right_only = set(right_times.keys()) - set(left_times.keys())
    paired = set(left_times.keys()) & set(right_times.keys())
    
    print("=" * 80)
    print("PAIRING RESULTS")
    print("=" * 80)
    print(f"Properly paired: {len(paired)}")
    print(f"Left only (no matching right): {len(left_only)}")
    print(f"Right only (no matching left): {len(right_only)}")
    print()
    
    if left_only:
        print(f"Left-only files (showing first 20):")
        for i, key in enumerate(sorted(left_only)[:20]):
            print(f"  {left_times[key]}")
        if len(left_only) > 20:
            print(f"  ... and {len(left_only) - 20} more")
        print()
    
    if right_only:
        print(f"Right-only files (showing first 20):")
        for i, key in enumerate(sorted(right_only)[:20]):
            print(f"  {right_times[key]}")
        if len(right_only) > 20:
            print(f"  ... and {len(right_only) - 20} more")
        print()
    
    # Check if pairing is good enough
    total_files = len(left_times) + len(right_times)
    pairing_rate = (len(paired) * 2) / total_files * 100 if total_files > 0 else 0
    
    print("=" * 80)
    print(f"Pairing rate: {pairing_rate:.2f}%")
    
    if pairing_rate > 99.9:
        print("✓ Excellent pairing! Almost all files have matching pairs.")
    elif pairing_rate > 99:
        print("✓ Good pairing! Most files have matching pairs.")
    elif pairing_rate > 95:
        print("⚠ Acceptable pairing, but some files are unpaired.")
    else:
        print("✗ Poor pairing! Many files don't have matching pairs.")
    
    print("=" * 80)
    
    return len(left_only) == 0 and len(right_only) == 0

if __name__ == "__main__":
    left_folder = r"F:\Image storage\Pattern1_left"
    right_folder = r"F:\Image storage\Pattern1_right"
    
    check_pairing(left_folder, right_folder)
