import os
import zipfile
import shutil
from pydub import AudioSegment

# Hardcoded path to the ZIP file
ZIP_FILE_PATH = r"C:\Users\aksha\Deep Learning Labs\MHA Project\Hindi_Audio_Datasets\archive\temp.zip"

# Define directories
EXTRACT_TO = r"C:\Users\aksha\Deep Learning Labs\MHA Project\Hindi_Audio_Datasets\archive\extracted_audio"
OUTPUT_FOLDER = r"C:\Users\aksha\Deep Learning Labs\MHA Project\Hindi_Audio_Datasets\archive\processed_audio"

def extract_zip(zip_path, extract_to):
    """Extracts the ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted ZIP to: {extract_to}")

def split_audio(audio_path, output_folder, chunk_duration=30, overlap=3):
    """Splits an audio file into overlapping 30-second chunks."""
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000  # Convert milliseconds to seconds

    if duration_sec <= chunk_duration:
        print(f"Skipping split, {os.path.basename(audio_path)} is already <= {chunk_duration} sec")
        return

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    start_time = 0
    chunk_num = 1

    while start_time < duration_sec:
        end_time = min(start_time + chunk_duration, duration_sec)
        chunk = audio[start_time * 1000 : end_time * 1000]
        
        chunk_filename = f"{file_name}_chunk{chunk_num}.wav"
        chunk_output_path = os.path.join(output_folder, chunk_filename)
        chunk.export(chunk_output_path, format="wav")

        print(f"Saved: {chunk_output_path}")

        start_time += (chunk_duration - overlap)
        chunk_num += 1

def process_audio_files(zip_path):
    """Extracts ZIP and processes audio files."""
    
    # Clean previous extractions
    if os.path.exists(EXTRACT_TO):
        shutil.rmtree(EXTRACT_TO)
    os.makedirs(EXTRACT_TO, exist_ok=True)

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Step 1: Extract ZIP
    extract_zip(zip_path, EXTRACT_TO)

    # Step 2: Process each audio file
    for root, _, files in os.walk(EXTRACT_TO):
        for file in files:
            if file.endswith((".wav", ".mp3", ".flac")):  # Add more formats if needed
                audio_path = os.path.join(root, file)
                split_audio(audio_path, OUTPUT_FOLDER)

if __name__ == "__main__":
    process_audio_files(ZIP_FILE_PATH)
