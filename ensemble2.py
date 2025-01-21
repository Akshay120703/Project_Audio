from collections import Counter
import re

# Load transcriptions from files
def load_transcriptions(file_path: str) -> dict:
    transcriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify utf-8 encoding
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    audio_id, transcription = line.strip().split(None, 1)
                    transcriptions[audio_id] = transcription
                except ValueError:
                    continue  # Skip lines that do not match the expected format
    return transcriptions

# Combine transcriptions for each audio file
def combine_transcriptions(transcriptions_list: list) -> dict:
    combined_results = {}
    for audio_id in transcriptions_list[0].keys():
        all_transcriptions = [t[audio_id] for t in transcriptions_list if audio_id in t]
        longest_transcription = max(all_transcriptions, key=len, default="")  # Longest transcription
        combined_results[audio_id] = longest_transcription
    return combined_results

# Save combined transcriptions to a file
def save_combined_transcriptions(combined_results: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as file:  # Specify utf-8 encoding
        for audio_id, transcription in combined_results.items():
            file.write(f"{audio_id} {transcription}\n")

if __name__ == "__main__":
    # Example input files
    file1 = "file2.txt"
    file2 = "file3.txt"
    file3 = "file4.txt"
    file4 = "file5.txt"
    
    # Load transcriptions from each ASR model's output
    transcriptions1 = load_transcriptions(file1)
    transcriptions2 = load_transcriptions(file2)
    transcriptions3 = load_transcriptions(file3)
    transcriptions4 = load_transcriptions(file4)    
    # Combine transcriptions
    combined_transcriptions = combine_transcriptions([transcriptions1, transcriptions2, transcriptions3, transcriptions4])

    # Save to an output file
    output_file = "combined_transcriptions.txt"
    save_combined_transcriptions(combined_transcriptions, output_file)

    print(f"Combined transcriptions saved to {output_file}")
