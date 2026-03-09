import glob
import os

vid_root = "./dataset/Phoenix14T"
# Using the exact folder name we saw in your tree
pattern = os.path.join(vid_root, 'features', 'fullFrame-210x260px', '*', '25October_2010_Monday_tagesschau-17', '*.png')

print(f"Checking pattern: {pattern}")
files = glob.glob(pattern)
print(f"Files found: {len(files)}")

if len(files) > 0:
    print(f"Sample file: {files[0]}")
else:
    # Try a broader search to find where the mistake is
    print("Zero files found. Checking directory existence...")
    check_dir = os.path.join(vid_root, 'features', 'fullFrame-210x260px', 'test', '25October_2010_Monday_tagesschau-17')
    print(f"Does the specific test folder exist? {os.path.exists(check_dir)}")