import whisper
import json
from pathlib import Path
import os
import subprocess

# Set FFmpeg path manually
ffmpeg_path = r"C:\FFmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Check if FFmpeg is installed correctly
def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed correctly.")
        else:
            print("❌ FFmpeg check failed. Ensure it is installed in:", ffmpeg_path)
            exit(1)
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install it and check the path:", ffmpeg_path)
        exit(1)

check_ffmpeg()


# Step 1: Find Media Files
def find_media_files(directory):
    """
    Recursively scans the directory and returns a list of media files.
    """
    media_extensions = {".mp3", ".wav", ".flac", ".mp4", ".mkv", ".mov"}
    return [str(file) for file in Path(directory).rglob("*") if file.suffix.lower() in media_extensions]


# Step 2: Transcribe Using Whisper
def transcribe_audio(file_path, model):
    """
    Uses OpenAI's Whisper model to transcribe an audio/video file.
    """
    try:
        file_path = Path(file_path).resolve()
        print(f"🎤 Processing: {file_path.name} ...")
        result = model.transcribe(str(file_path), language="en")
        return result["text"]
    except Exception as e:
        print(f"❌ Error transcribing {file_path.name}: {e}")
        return None


# Step 3: Save Transcriptions in JSON Format
def save_transcription(file_path, transcript, output_dir="transcriptions"):
    """
    Saves the transcription in JSON format in an output directory.
    """
    if transcript is None:
        return  # Skip saving if transcription failed

    try:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{file_path.stem}.json"

        with output_file.open("w", encoding="utf-8") as f:
            json.dump({"file": str(file_path.name), "transcription": transcript}, f, indent=4)

        print(f"✅ Saved transcription: {output_file}")
    except Exception as e:
        print(f"❌ Error saving transcription for {file_path.name}: {e}")


# Step 4: Process the Directory
def process_directory(directory):
    """
    Processes all media files in the given directory.
    """
    media_files = find_media_files(directory)

    if not media_files:
        print("❌ No media files found.")
        return

    print(f"🔍 Found {len(media_files)} media files. Loading model...")

    model = whisper.load_model("base")  # Change model to 'tiny', 'small', etc. if needed

    for file in media_files:
        transcript = transcribe_audio(file, model)
        save_transcription(file, transcript)

    print("\n✅ Transcription completed! Check the 'transcriptions/' folder.")


# Step 5: Run the Script
if __name__ == "__main__":
    folder_path = "."  # Change this to your media folder path
    process_directory(folder_path)
