import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel
import decord
from threading import Thread
from queue import Queue

import sys
sys.path.append('./')
import gc
import signal

stop_requested = False
def sigint_handler(signum, frame):
    global stop_requested
    print("\n[Ctrl+C detected] Will exit gracefully after finishing the current video...")
    stop_requested = True

signal.signal(signal.SIGINT, sigint_handler)


from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)



class ViTFeatureReader(object):
    def __init__(
        self, 
        model_name='openai/clip-vit-large-patch14', 
        cache_dir=None,
        device='cuda:0', 
        s2_mode='s2wrapping',
        scales=[1, 2],
        nth_layer=-1
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer
        
        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(list(video), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--s2_mode', default='')
    parser.add_argument('--scales', nargs='+', type=int, help='List of scales', default=[])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    parser.add_argument('--shard_id', type=int, default=0, help='shard ID for parallelization')
    parser.add_argument('--num_shards', type=int, default=1, help='total shards for parallelization')
    
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='openai/clip-vit-large-patch14')
    parser.add_argument('--reverse', action='store_true', help='Process segments in reverse order')
    return parser

def get_iterator(args, mode):
    batch_size = args.batch_size
    
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    
    # Apply sharding
    keys = list(data.keys())
    # The last key might be a special key or just string integers, sort them just in case
    # data is dict of str(int) -> dict
    # filter out non-digit keys if any (like 'original_info' etc if stored globally, though usually not)
    keys = sorted([k for k in keys if k.isdigit()], key=lambda x: int(x))
    
    sharded_keys = keys[args.shard_id::args.num_shards]
    if args.reverse:
        sharded_keys = sharded_keys[::-1]
    num = len(sharded_keys)
    
    ds_name = osp.split(args.anno_root)[-1]
    reader = ViTFeatureReader(
        args.model_name, 
        device=args.device, 
        s2_mode=args.s2_mode, 
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    def iterate(save_path, postfix_template):
        # Using decord for much faster loading
        def load_video_decord(path):
            try:
                vr = decord.VideoReader(path, num_threads=4)
                # Decode all frames as a batch
                frames = vr.get_batch(range(len(vr))).asnumpy()
                # Convert to list of PIL Images for the processor
                return [Image.fromarray(f) for f in frames]
            except Exception as e:
                # print(f"Error loading {path} with decord: {e}")
                return read_video(path) # Fallback to PyAV

        def loader_worker(keys_list, queue):
            for k in keys_list:
                if stop_requested:
                    break
                    
                item = data[k]
                fid = item['fileid']
                fname = osp.join(args.video_root, item['folder'])
                
                # Check if video exists to avoid FileNotFoundError
                if not osp.exists(fname):
                    print(f"Soft-Skip: Video file not found: {fname}")
                    queue.put(("SKIPPED", fid, None))
                    continue

                # Check skip before loading to save CPU
                target_file = osp.join(save_path, f'{fid}{postfix_template}.npy')
                if osp.exists(target_file):
                    queue.put(("SKIPPED", fid, None))
                    continue

                if ds_name == 'Phoenix14T' or ds_name == 'CSL-Daily':
                    image_list = get_img_list(ds_name, args.video_root, fname)
                    videos = [Image.open(image).convert('RGB') for image in image_list]
                else:
                    videos = load_video_decord(fname)
                
                queue.put((videos, fid, None))
            queue.put(None) # Sentinel

        # Start pre-fetching thread - reduced to 2 to minimize RAM spikes
        fetch_queue = Queue(maxsize=2)
        worker = Thread(target=loader_worker, args=(sharded_keys, fetch_queue))
        worker.start()

        while True:
            msg = fetch_queue.get()
            if msg is None:
                break
            
            videos, fid, _ = msg
            
            if videos == "SKIPPED":
                yield "SKIPPED", fid, None
                continue

            if len(videos) > 0:
                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)
                yield np.concatenate(video_feats, axis=0), fid, None
                # Explicitly clear large memory blocks
                del videos
                del video_feats
                gc.collect()
            else:
                yield [], fid, None
        
        worker.join()
    
    return iterate, num

def main():
    mode = ["dev", "test", "train"]
    for m in mode:
        parser = get_parser()
        args = parser.parse_args()

        ds_name = osp.split(args.anno_root)[-1]
        _model_name = os.path.split(args.model_name)[-1]
        fname = f'{_model_name}_feat_{ds_name}'
        
        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)
    
        if ds_name == 'How2Sign':
            _m = m
        elif ds_name == 'NIASL2021':
            if m == 'dev': _m = 'validation' 
        else:
            _m = m

        save_path_base = osp.join(args.save_dir, fname, m)
        postfix_template = ""
        if args.s2_mode != "":
            postfix_template = f"_{args.s2_mode}"
        if len(args.scales) == 3:
            postfix_template = f'{postfix_template}_large'

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
if __name__ == "__main__":
    main()