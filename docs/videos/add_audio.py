import os
import subprocess
import tempfile

def list_videos(folder_path):
    """List all video files in the specified folder."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    videos = [f for f in os.listdir(folder_path) 
              if os.path.isfile(os.path.join(folder_path, f)) and 
              os.path.splitext(f)[1].lower() in video_extensions]
    return videos

def swap_audio(source_video, target_video):
    """Swap audio from source_video to target_video using ffmpeg and replace original."""
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Build ffmpeg command to process to temporary file
    cmd = [
        'ffmpeg',
        '-i', target_video,          # Input target video (video stream)
        '-i', source_video,          # Input source video (audio stream)
        '-c:v', 'copy',              # Copy video codec (no re-encoding)
        '-c:a', 'aac',               # Encode audio to AAC (for compatibility)
        '-map', '0:v:0',             # Use video from first input (target)
        '-map', '1:a:0',             # Use audio from second input (source)
        '-shortest',                 # Trim to shortest stream length
        '-y',                        # Overwrite output file without prompt
        temp_path
    ]
    
    try:
        # Run ffmpeg command
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Replace original file with processed file
        os.replace(temp_path, target_video)
        print(f"Successfully processed: {target_video}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {target_video}: {e}")
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    except Exception as e:
        print(f"Unexpected error with {target_video}: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def main():
    folder_path = '.'  # Current directory; change if needed
    source_video_name = 'source.mp4'
    
    # Verify source file exists
    source_path = os.path.join(folder_path, source_video_name)
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_video_name} not found!")
        return
    
    # List all videos
    videos = list_videos(folder_path)
    print(f"Found videos: {videos}")
    
    # Process each video except source.mp4
    for video in videos:
        if video != source_video_name:
            target_path = os.path.join(folder_path, video)
            swap_audio(source_path, target_path)

if __name__ == "__main__":
    main()