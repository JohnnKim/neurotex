import os
import re

# Number of most recent iterations to keep
KEEP_LAST_N = 5

def extract_iter_num(filename):
    """
    Extract the iteration number from a filename like 'neurotex_iter_01.pth'.
    Returns the integer iteration number if found, else None.
    """
    match = re.search(r"_iter_(\d{2})", filename)
    return int(match.group(1)) if match else None

def delete_old_versions():
    """
    Scans the current directory for versioned iteration files (e.g., *_iter_01.pth),
    and deletes all but the most recent N iterations based on KEEP_LAST_N.
    """
    # All versioned files
    files = os.listdir()
    # Filter only files that match the pattern *_iter_XX.*
    iter_files = [f for f in files if re.search(r"_iter_\d{2}", f)]

    # Map iteration number to list of associated filenames
    iter_map = {}
    for f in iter_files:
        num = extract_iter_num(f)
        if num is not None:
            iter_map.setdefault(num, []).append(f)

    # All found iteration numbers sorted
    all_iters = sorted(iter_map.keys())
    # Only keep the latest N
    keep_iters = set(all_iters[-KEEP_LAST_N:])

    deleted = 0
    for num in all_iters:
        if num in keep_iters:
            # Skip the ones we're keeping
            continue
        for f in iter_map[num]:
            # Delete old file
            os.remove(f)
            print(f"Deleted: {f}")
            deleted += 1

    if deleted == 0:
        print(f"Nothing to delete. Only {len(all_iters)} iteration(s) found.")
    else:
        print(f"Cleanup complete. Deleted {deleted} file(s). Kept last {KEEP_LAST_N} iterations.")

if __name__ == "__main__":
    delete_old_versions()
