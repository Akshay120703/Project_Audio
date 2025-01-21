from collections import Counter
import re

# Define a function to detect hallucinations
def is_hallucinated(transcription: str, threshold: float = 0.8) -> bool:
    words = transcription.split()
    word_count = len(words)
    if word_count == 0:
        return True

    # Count occurrences of 1-word to 5-word phrases
    hallucination_patterns = []
    for n in range(1, 6):
        phrases = [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]
        counter = Counter(phrases)
        most_common_phrase, frequency = counter.most_common(1)[0]

        # Check if the most common phrase is overly frequent
        if frequency / (word_count - n + 1) > threshold:
            return True
    return False

# Load transcriptions from files
def load_transcriptions(file_path: str) -> dict:
    transcriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:  # Specify utf-8 encoding
        for line in file:
            audio_id, transcription = line.strip().split(None, 1)
            transcriptions[audio_id] = transcription
    return transcriptions

# Combine transcriptions for each audio file
def combine_transcriptions(transcriptions_list: list) -> dict:
    combined_results = {}
    for audio_id in transcriptions_list[0].keys():
        all_transcriptions = [t[audio_id] for t in transcriptions_list if audio_id in t]
        valid_transcriptions = sorted(
            [t for t in all_transcriptions if not is_hallucinated(t)],
            key=len,
            reverse=True
        )
        
        if valid_transcriptions:
            combined_results[audio_id] = valid_transcriptions[0]  # Select the longest valid transcription
        else:
            combined_results[audio_id] = ""  # Fallback if all are hallucinated
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
