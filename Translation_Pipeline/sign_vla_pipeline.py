import os
import sys
import argparse
import socket
import json
import time
import shutil
import tempfile

import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from translate_video import (
    extract_and_crop_frames,
    extract_spatial_features,
    extract_motion_features,
    load_spamo_model,
    translate,
)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(
        script_dir, '..', 'logs',
        '2026-03-27T23-05-35_spamo_how2sign', 'checkpoints', 'last.ckpt'
    )
    default_cfg = os.path.join(script_dir, '..', 'configs', 'finetune_how2sign.yaml')
    default_out = os.path.join(script_dir, 'translation_target')

    parser = argparse.ArgumentParser(description="SpaMo -> GR00T N1 VLA pipeline")
    parser.add_argument('--ckpt_path', type=str, default=default_ckpt)
    parser.add_argument('--config_path', type=str, default=default_cfg)
    parser.add_argument('--duration', type=int, default=5,
                        help="Auto-stop recording after N seconds. 0 = stop on Q press.")
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--confirm', action='store_true',
                        help="Prompt for confirmation before sending to GR00T N1.")
    parser.add_argument('--groot_host', type=str, default='localhost')
    parser.add_argument('--groot_port', type=int, default=9999)
    parser.add_argument('--output_dir', type=str, default=default_out)
    return parser.parse_args()


def record_video(output_path, fps, duration):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("=" * 50)
    print("RECORDING STARTED")
    print(f"Saving to: {output_path}")
    if duration > 0:
        print(f"Auto-stops after {duration}s. Press 'q' to stop early.")
    else:
        print("Press 'q' on the video window to stop recording.")
    print("=" * 50)

    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: cannot read frame from webcam.")
                break

            out.write(frame)

            display_frame = frame.copy()
            if duration > 0:
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                cv2.putText(
                    display_frame,
                    f"Recording: {remaining:.1f}s remaining",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow('Recording (press q to stop)', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            if duration > 0 and (time.time() - start_time) >= duration:
                break
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Video saved: {output_path}")


def send_to_groot(command_text, host, port):
    payload = json.dumps({"command": command_text, "timestamp": time.time()}).encode('utf-8')
    try:
        # TODO: replace with actual GR00T N1 API call if interface changes
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            response = b''.join(chunks).decode('utf-8')
            return response
    except Exception as e:
        print(f"[GR00T N1] Send failed: {e}")
        return None


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("SpaMo -> GR00T N1 VLA Pipeline")
    print(f"Device: {DEVICE}")
    print("Loading SpaMo model...")
    spamo_model = load_spamo_model(args.ckpt_path, args.config_path)
    print("SpaMo model loaded. Ready.")
    print("=" * 50)

    while True:
        user_input = input("\nPress ENTER to record, 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("Exiting.")
            break

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(args.output_dir, f"recording_{timestamp}.mp4")

        record_video(video_path, args.fps, args.duration)

        temp_dir = tempfile.mkdtemp(prefix="sign_vla_")
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        try:
            print("Extracting frames...")
            frames = extract_and_crop_frames(video_path, frames_dir)
            print(f"{len(frames)} frames extracted.")

            spatial_feats = extract_spatial_features(frames)
            print(f"Spatial features: {spatial_feats.shape}")

            motion_feats = extract_motion_features(frames)
            print(f"Motion features: {motion_feats.shape}")

            translation = translate(spamo_model, spatial_feats, motion_feats)

            print("\n" + "=" * 50)
            print("TRANSLATION:")
            print(f"  {translation}")
            print("=" * 50)

            if args.confirm:
                confirm_input = input("Send this command to GR00T N1? [y/N]: ").strip().lower()
                if confirm_input != 'y':
                    print("Command not sent.")
                    continue

            print(f"Sending to GR00T N1 at {args.groot_host}:{args.groot_port}...")
            response = send_to_groot(translation, args.groot_host, args.groot_port)
            if response is not None:
                print(f"[GR00T N1] Response: {response}")
            else:
                print("[GR00T N1] No response received.")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
