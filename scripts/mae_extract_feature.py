import os
import numpy as np
import torch
import argparse
import tqdm
import os.path as osp
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
sys.path.append('./')

import signal

stop_requested = False
def sigint_handler(signum, frame):
    global stop_requested
    print("\n[Ctrl+C detected] Will exit gracefully after finishing the current video...")
    stop_requested = True

signal.signal(signal.SIGINT, sigint_handler)


from utils.helpers import sliding_window_for_list, read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class VideoMAEFeatureReader(object):
    def __init__(
        self, 
        model_name='MCG-NJU/videomae-large', 
        cache_dir=None,
        device='cuda:0',
        overlap_size=0,
        nth_layer=-1
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()
        
    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        
        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]
        
        return outputs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    parser.add_argument('--shard_id', type=int, default=0, help='shard ID for parallelization')
    parser.add_argument('--num_shards', type=int, default=1, help='total shards for parallelization')
    return parser


def get_iterator(args, mode):
    batch_size = args.batch_size

    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    
    # Apply sharding
    keys = list(data.keys())
    # filter out non-digit keys if any
    keys = sorted([k for k in keys if k.isdigit()], key=lambda x: int(x))
    
    sharded_keys = keys[args.shard_id::args.num_shards]
    num = len(sharded_keys)
    
    ds_name = osp.split(args.anno_root)[-1]

    reader = VideoMAEFeatureReader(
        args.model_name, 
        device=args.device, 
        overlap_size=args.overlap_size, 
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    def iterate(save_path, postfix_template):
        for k in sharded_keys:
            item = data[k]
            fid = item['fileid']
            
            # --- GLOBAL RESUME LOGIC ---
            # Check if file exists before processing anything
            target_file = osp.join(save_path, f'{fid}{postfix_template}.npy')
            if osp.exists(target_file):
                yield "SKIPPED", fid, None
                continue
            
            fname = osp.join(args.video_root, item['folder'])
            
            if ds_name == 'Phoenix14T' or ds_name == 'CSL-Daily':
                image_list = get_img_list(ds_name, args.video_root, fname)
                
                if len(image_list) < 16:
                    image_list.extend([image_list[-1]] * (16 - len(image_list)))
                image_list_chunks = sliding_window_for_list(image_list, window_size=16, overlap_size=args.overlap_size)
                
                videos = []
                for image_list in image_list_chunks:
                    videos.append([Image.open(image).convert('RGB') for image in image_list])
                
                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)
                    
                yield np.concatenate(video_feats, axis=0), fid, None
            
            else:
                # How2Sign and others
                videos = read_video(fname)
                
                if len(videos) > 0:
                    if len(videos) < 16:
                        videos.extend([videos[-1]] * (16 - len(videos)))
                    
                    videos = sliding_window_for_list(videos, window_size=16, overlap_size=args.overlap_size)
                    
                    video_feats = []
                    for j in range(0, len(videos), batch_size):
                        video_batch = videos[j:min(j + batch_size, len(videos))]
                        feats = reader.get_feats(video_batch).cpu().numpy()
                        video_feats.append(feats)
                    
                    yield np.concatenate(video_feats, axis=0), fid, None
                
                else:
                    yield [], fid, None
    
    return iterate, num

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Determine modes to process
    modes_to_process = args.mode if args.mode else ["dev", "test", "train"]
    
    for m in modes_to_process:
        ds_name = osp.split(args.anno_root)[-1]
        fname = f'mae_feat_{ds_name}'
        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)
    
        if ds_name == 'How2Sign':
            _m = m
        elif ds_name == 'NIASL2021':
            _m = 'validation' if m == 'dev' else m
        else:
            _m = m

        save_path_base = osp.join(args.save_dir, fname, m)
        postfix_template = f'_overlap-{args.overlap_size}'

        generator, num = get_iterator(args, _m)
        iterator = generator(save_path_base, postfix_template)

        print(f"Starting/Resuming processing for mode: {m}")
        for vit_feat in tqdm.tqdm(iterator, total=num):
            feats, id, st = vit_feat
            
            # Handle the skip signal
            if isinstance(feats, str) and feats == "SKIPPED":
                continue

            postfix = postfix_template
            if st is not None:
                postfix = f'_{st}{postfix}'
            
            np.save(osp.join(save_path_base, f'{id}{postfix}.npy'), feats)

            if stop_requested:
                print("Graceful exit requested. Closing.")
                sys.exit(0)


if __name__ == "__main__":
    main()