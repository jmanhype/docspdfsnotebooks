# script/faster_whisper_translation.py

import logging
import subprocess
import os
from faster_whisper import WhisperModel
import tempfile

# Configure logging
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

# Provided file paths
video_file_path = "C:\\Users\\strau\\Documents\\Obsidian Vault\\cena.mp4"
output_file_path = "C:\\Users\\strau\\Documents\\Obsidian Vault\\translation_output.txt"

# Initialize the WhisperModel for translation
transcriber = WhisperModel("large-v2", device="cuda", compute_type="float16")

# Function to extract audio from video using FFmpeg and return audio file path
def extract_audio_with_ffmpeg(video_path):
    try:
        temp_audio_path = tempfile.mktemp(suffix='.mp3')
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'mp3', temp_audio_path]
        subprocess.run(cmd, check=True)
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None

# Function to translate audio and save to a file
def translate_and_save(transcriber, audio_path, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            segments, info = transcriber.transcribe(audio_path, task="translate")

            # Concatenate all translated segments
            full_translation = ' '.join(segment.text for segment in segments)

            # Write the complete translation to the file
            file.write(full_translation)
    except Exception as e:
        logging.error(f"Error during translation: {e}")

# Main workflow
def main():
    audio_file_path = extract_audio_with_ffmpeg(video_file_path)
    if audio_file_path:
        translate_and_save(transcriber, audio_file_path, output_file_path)
        os.remove(audio_file_path)  # Clean up temporary audio file
    else:
        logging.error("Failed to extract audio from video.")

if __name__ == "__main__":
    main()

