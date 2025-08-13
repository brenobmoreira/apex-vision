import cv2
import os
from pathlib import Path

def extract_frames_interactive(video_path):
    """
    Interactive video player for frame extraction.
    
    Controls:
    - SPACE: Pause/Resume
    - 's': Save current frame
    - 'a'/'d': Previous/Next frame (when paused)
    - 'q'/'ESC': Quit
    """
    
    # Create output directory
    output_dir = Path("extracted_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video loaded: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  's': Save current frame")
    print("  'a'/'d': Previous/Next frame (when paused)")
    print("  'q'/ESC: Quit")
    
    paused = False
    frame_count = 0
    saved_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        cv2.imshow('Video - Press SPACE to pause, S to save frame', frame)
        
        delay = int(1000 / fps) if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'} at frame {frame_count}")
            
        elif key == ord('s'):
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
            print(f"Saved frame {frame_count} as {filename}")
            
        elif key == ord('a') and paused:
            new_frame = max(0, frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = cap.read()
            if ret:
                frame_count = new_frame
                
        elif key == ord('d') and paused:
            ret, frame = cap.read()
            if ret:
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                print("End of video reached")
                break
                
        elif key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nExtraction complete! Saved {saved_count} frames to '{output_dir}' folder")

def main():
    # Check for video file
    video_files = ['brocas.mp4']
    video_path = None
    
    for file in video_files:
        if os.path.exists(file):
            video_path = file
            break
    
    if not video_path:
        print("No video file found. Please ensure you have:")
        for file in video_files:
            print(f"  - {file}")
        return
    
    extract_frames_interactive(video_path)

if __name__ == "__main__":
    main()