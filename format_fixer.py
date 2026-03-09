import os
from pathlib import Path

# The official root structure
base_path = Path("./dataset/Phoenix14T/features/fullFrame-210x260px")

for split in ["dev", "test", "train"]:
    split_path = base_path / split
    if not split_path.exists(): continue
    
    print(f"🔗 Processing {split}...")
    for folder in split_path.iterdir():
        if folder.is_dir():
            images = sorted(list(folder.glob("images*.png")))
            for img in images:
                # Extract number for SpaMo format (images0001.png -> 1.png)
                num = img.stem.replace("images", "").lstrip("0")
                if not num: num = "0"
                
                link = folder / f"{num}.png"
                if not link.exists():
                    os.symlink(img.name, link)