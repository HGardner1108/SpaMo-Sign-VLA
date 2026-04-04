import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def preprocess_how2sign(csv_path: str, split: str, save_dir: str):
    """
    Convert How2Sign CSV into SpaMo .npy annotation format.
    Expects CSVs like how2sign_realigned_train.csv
    """
    df = pd.read_csv(csv_path, sep='\t')
    
    # Map split names to match How2Sign directory structure
    if split == 'dev':
        h2s_split = 'val'
    else:
        h2s_split = split
        
    annotations = {}
    
    # Filter out missing translations if any
    df = df.dropna(subset=['SENTENCE'])
    
    for idx, row in df.iterrows():
        sentence_id = row['SENTENCE_NAME']
        
        # For the HF clips data (bdanko/how2sign-rgb-front-clips), 
        # the .mp4 files are exactly named as the sentence_id
        folder_path = f"{h2s_split}/rgb_front/{sentence_id}.mp4"
        
        
        annotations[str(idx)] = {
            'fileid': sentence_id,
            'folder': folder_path,
            'text': row['SENTENCE'],
            'gloss': '', # How2Sign has glosses in a different file, but not strictly needed for this baseline
            'lang': 'English',
            'num_frames': -1, # Will be determined during feature extraction or loading
            'original_info': row.to_dict(),
            'en_text': row['SENTENCE'],
            'fr_text': '',
            'es_text': '',
            'tag': 'how2sign'
        }
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save generic and ML versions (SpaMo p14t.py loads _info_ml.npy)
    np.save(os.path.join(save_dir, f'{split}_info.npy'), annotations)
    np.save(os.path.join(save_dir, f'{split}_info_ml.npy'), annotations)
    
    print(f"[{split}] Processed {len(annotations)} samples -> {save_dir}/{split}_info_ml.npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory (e.g. ./How2Sign/sentence_level)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save .npy files')
    args = parser.parse_args()
    
    # How2Sign standard dump names from download script
    splits = {
        'train': ('how2sign_realigned_train.csv', 'train'),
        'dev': ('how2sign_realigned_val.csv', 'val'),
        'test': ('how2sign_realigned_test.csv', 'test')
    }
    
    for split, (filename, folder_name) in splits.items():
        csv_path = os.path.join(args.base_dir, folder_name, 'text/en/raw_text/re_aligned', filename)
        if os.path.exists(csv_path):
            preprocess_how2sign(csv_path, split, args.save_dir)
        else:
            print(f"Warning: {csv_path} not found.")
