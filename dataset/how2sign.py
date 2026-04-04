import torch
import os
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import random

class How2Sign(torch.utils.data.Dataset):
    """
    Dataset class for the How2Sign english sign language dataset.
    Adapted from Phoenix14T.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = 'dev',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = ''
    ):
        super().__init__()
        
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        anno_path = self.anno_root / f'{mode}_info_ml.npy'
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
            
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        
        self._validate_directories()
        
        # Filter data to only include samples with existing features
        filtered_data = {}
        missing_count = 0
        
        for idx_str, sample in self.data.items():
            file_id = sample['fileid']
            exists = True
            
            # Check spatial
            if self.spatial:
                spatial_exists = False
                for ext in ['.pt', '.npy']:
                    if (self.spatial_dir / f"{file_id}{self.spatial_postfix}{ext}").exists():
                        spatial_exists = True
                        break
                if not spatial_exists:
                    exists = False
            
            # Check spatiotemporal
            if exists and self.spatiotemporal:
                if isinstance(self.spatiotemporal_postfix, str):
                    if not (self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy").exists():
                        exists = False
                else:
                    for postfix in self.spatiotemporal_postfix:
                        if not (self.spatiotemporal_dir / f"{file_id}{postfix}.npy").exists():
                            exists = False
                            break
            
            if exists:
                # Keep index as string but re-index or just keep original?
                # The data loading uses str(index) where index is from 0 to len().
                # So we MUST re-index to prevent KeyError.
                filtered_data[str(len(filtered_data))] = sample
            else:
                missing_count += 1
        
        self.data = filtered_data
        print(f"[INFO] How2Sign dataset (mode={mode}): Kept {len(self.data)} samples, skipped {missing_count} missing samples.", flush=True)

    def _validate_directories(self) -> None:
        if self.spatial and not self.spatial_dir.exists():
            print(f"Warning: Spatial feature directory not found: {self.spatial_dir}. Expected if running feature extraction or fast tests.")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            print(f"Warning: Spatiotemporal feature directory not found: {self.spatiotemporal_dir}. Expected if running feature extraction or fast tests.")

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        # Try .pt first (from our ViT extraction) then .npy
        for ext in ['.pt', '.npy']:
            feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}{ext}"
            if feat_path.exists():
                if ext == '.pt':
                    return torch.load(feat_path, map_location='cpu')
                else:
                    return torch.tensor(np.load(feat_path), dtype=torch.float32)
        
        raise FileNotFoundError(f"Spatial feature file not found for {file_id} with postfix {self.spatial_postfix} in {self.spatial_dir}")

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.spatiotemporal_postfix, str):
            feat_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not feat_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {feat_path}")
            return torch.tensor(np.load(feat_path), dtype=torch.float32)
        else:
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path), dtype=torch.float32))
            return features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Dictionary keys are strings from the pandas dump
        data = self.data[str(index)]
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                # print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                # print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]
        
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': self._normalize_text(data['text']),
            'gloss': data.get('gloss', ''),
            'id': file_id,
            'num_frames': len(pixel_value) if pixel_value is not None and len(pixel_value) > 0 else 0,
            'vid_path': str(self.vid_root / data['folder']),
            'lang': 'English',
            'en_text': self._normalize_text(data['text']),
            'fr_text': '',
            'es_text': ''
        }
        
        result['original_info'] = data.get('original_info', {})
        
        return result

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
