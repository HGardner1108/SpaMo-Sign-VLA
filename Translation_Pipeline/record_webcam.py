import cv2
import os
import argparse
import time

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(script_dir, 'translation_target')

    parser = argparse.ArgumentParser(description="Record webcam video and save to translation_target/")
    parser.add_argument('--output_dir', type=str, default=default_out, 
                        help="Directory to save the recorded video")
    parser.add_argument('--fps', type=int, default=25, help="Target frames per second for recording")
    # Phoenix14T uses 25 FPS
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if there are existing files in translation_target and warn user
    existing_files = os.listdir(args.output_dir)
    if existing_files:
        print(f"Warning: The directory '{args.output_dir}' already contains {len(existing_files)} file(s).")
        print("You might want to clear them before running the translation pipeline.")

    # Format output filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(args.output_dir, f"recording_{timestamp}.mp4")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get default resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera opened. Resolution: {frame_width}x{frame_height}")

    # Define the codec and create VideoWriter object
    # Using mp4v codec for standard mp4 compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, args.fps, (frame_width, frame_height))

    print("="*50)
    print("🎥 RECORDING STARTED")
    print(f"Saving to: {output_path}")
    print("Press 'q' or 'ESC' on the video window to STOP recording.")
    print("="*50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Write the frame to the output file
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Webcam Recording (Press q to stop)', frame)

            # Break the loop on 'q' or ESC key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        print("\nRecording stopped by user interrupt.")
    finally:
        # Release everything when job is finished
        print("Saving video file...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Video successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
