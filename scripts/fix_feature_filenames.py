import os
import re
from pathlib import Path

def fix_filenames(directory_path: str):
    root_dir = Path(directory_path)
    if not root_dir.exists():
        print(f"Directory not found: {directory_path}")
        return

    pattern = re.compile(r'-rgb_front_[0-9.]+_')
    
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.npy'):
                continue
                
            match = pattern.search(file)
            if match:
                # Replace '-rgb_front_TIMESTAMP_' with '-rgb_front_'
                new_name = pattern.sub('-rgb_front_', file)
                
                old_path = os.path.join(subdir, file)
                new_path = os.path.join(subdir, new_name)
                
                # Check for overlap, in case the target filename already exists (e.g., duplicate extraction!)
                if os.path.exists(new_path) and old_path != new_path:
                    print(f"Warning: {new_path} already exists. Removing older {old_path} duplicate...")
                    os.remove(old_path)
                else:
                    os.rename(old_path, new_path)
                    count += 1
                
    print(f"Successfully renamed {count} files in {directory_path}.")

if __name__ == '__main__':
    print("Fixing Spatial Features...")
    fix_filenames('./features/spatial/clip-vit-large-patch14_feat_How2Sign')
    
    print("\nFixing Motion Features...")
    fix_filenames('./features/motion/mae_feat_How2Sign')
