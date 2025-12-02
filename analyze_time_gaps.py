import os
from pathlib import Path
from datetime import datetime
import re

def parse_filename_time(filename):
    """
    Extract datetime from filename like '2025-11-18 03-04-40.811；left；...'
    Returns datetime object or None
    """
    # Pattern: YYYY-MM-DD HH-MM-SS.mmm
    pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2})-(\d{2})-(\d{2})\.(\d{3})'
    match = re.match(pattern, filename)
    
    if match:
        date_str = match.group(1)  # 2025-11-18
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)
        millisecond = match.group(5)
        
        # Reconstruct as standard datetime string
        datetime_str = f"{date_str} {hour}:{minute}:{second}.{millisecond}"
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            return None
    return None

def analyze_time_gaps(folder_path, gap_threshold_seconds=60):
    """
    Analyze time gaps in filenames and find where sessions break
    
    Args:
        folder_path: Path to folder with timestamped files
        gap_threshold_seconds: Gap size to consider as session break (default 60 seconds)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    print(f"Analyzing time gaps in: {folder}")
    print(f"Gap threshold: {gap_threshold_seconds} seconds")
    print("=" * 80)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Total files found: {len(files)}")
    
    # Parse times and sort
    file_times = []
    for file in files:
        dt = parse_filename_time(file.name)
        if dt:
            file_times.append((dt, file.name))
    
    if not file_times:
        print("No valid timestamps found in filenames!")
        return
    
    # Sort by time
    file_times.sort(key=lambda x: x[0])
    
    print(f"Files with valid timestamps: {len(file_times)}")
    print(f"\nFirst file: {file_times[0][1]}")
    print(f"Last file:  {file_times[-1][1]}")
    print(f"\nTime range: {file_times[0][0]} to {file_times[-1][0]}")
    
    # Find gaps
    print(f"\n{'='*80}")
    print(f"ANALYZING GAPS (threshold: {gap_threshold_seconds} seconds)")
    print(f"{'='*80}\n")
    
    gaps = []
    session_count = 1
    session_start = file_times[0][0]
    session_file_count = 1
    
    for i in range(1, len(file_times)):
        prev_time, prev_name = file_times[i-1]
        curr_time, curr_name = file_times[i]
        
        time_diff = (curr_time - prev_time).total_seconds()
        
        if time_diff > gap_threshold_seconds:
            gaps.append({
                'gap_seconds': time_diff,
                'gap_minutes': time_diff / 60,
                'before_file': prev_name,
                'before_time': prev_time,
                'after_file': curr_name,
                'after_time': curr_time,
                'session_num': session_count,
                'session_file_count': session_file_count
            })
            
            # New session starts
            session_count += 1
            session_start = curr_time
            session_file_count = 1
        else:
            session_file_count += 1
    
    # Print gap analysis
    if gaps:
        print(f"Found {len(gaps)} significant gaps (session breaks):\n")
        for idx, gap in enumerate(gaps, 1):
            print(f"Gap #{idx} - Session {gap['session_num']} ended ({gap['session_file_count']} files)")
            print(f"  Gap duration: {gap['gap_seconds']:.1f} seconds ({gap['gap_minutes']:.1f} minutes)")
            print(f"  Last file before gap: {gap['before_file']}")
            print(f"  Time: {gap['before_time']}")
            print(f"  First file after gap: {gap['after_file']}")
            print(f"  Time: {gap['after_time']}")
            print()
    else:
        print("No significant gaps found! All files are in one continuous session.")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total sessions detected: {session_count}")
    print(f"Total files: {len(file_times)}")
    print(f"Average files per session: {len(file_times) / session_count:.1f}")
    
    # Detailed session info
    print(f"\n{'='*80}")
    print(f"SESSION DETAILS")
    print(f"{'='*80}\n")
    
    session_num = 1
    session_start_idx = 0
    
    for gap in gaps:
        # Find end index for this session
        for idx, (dt, name) in enumerate(file_times):
            if name == gap['before_file']:
                session_end_idx = idx
                break
        
        session_files = file_times[session_start_idx:session_end_idx+1]
        print(f"Session {session_num}:")
        print(f"  Files: {len(session_files)}")
        print(f"  Start: {session_files[0][0]} - {session_files[0][1]}")
        print(f"  End:   {session_files[-1][0]} - {session_files[-1][1]}")
        duration = (session_files[-1][0] - session_files[0][0]).total_seconds()
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print()
        
        session_num += 1
        session_start_idx = session_end_idx + 1
    
    # Last session
    if session_start_idx < len(file_times):
        session_files = file_times[session_start_idx:]
        print(f"Session {session_num}:")
        print(f"  Files: {len(session_files)}")
        print(f"  Start: {session_files[0][0]} - {session_files[0][1]}")
        print(f"  End:   {session_files[-1][0]} - {session_files[-1][1]}")
        duration = (session_files[-1][0] - session_files[0][0]).total_seconds()
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

if __name__ == "__main__":
    folder_path = r"F:\Image storage\Pattern1_right"
    
    print("=" * 80)
    print("TIME GAP ANALYZER")
    print("=" * 80)
    print()
    
    # You can adjust the threshold here
    # 60 seconds = 1 minute gap to consider as session break
    # Increase if you want larger gaps, decrease for smaller gaps
    analyze_time_gaps(folder_path, gap_threshold_seconds=60)
