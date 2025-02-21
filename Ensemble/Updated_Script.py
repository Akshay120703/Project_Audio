from collections import Counter

# Load transcriptions from files
def load_transcriptions(file_path: str) -> dict:
    transcriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    audio_id, transcription = line.strip().split(None, 1)
                    transcriptions[audio_id] = transcription
                except ValueError:
                    continue  # Skip lines that do not match the expected format
    return transcriptions

# Detect and remove hallucinations (phrases of 1-30 characters repeated more than 7 times)
def remove_hallucinations_strict(transcription: str) -> str:
    ngram_counts = Counter()

    # Generate and count substrings of 1 to 30 characters
    for start in range(len(transcription)):
        for length in range(1, 31):  # 1-character to 30-character sequences
            if start + length <= len(transcription):
                substring = transcription[start:start + length]
                ngram_counts[substring] += 1

    # Identify hallucinated phrases (repeated more than 7 times)
    hallucinated_phrases = {ngram for ngram, count in ngram_counts.items() if count > 7}

    # Remove all hallucinated phrases
    cleaned_transcription = transcription
    for phrase in hallucinated_phrases:
        cleaned_transcription = cleaned_transcription.replace(phrase, "")

    # Clean up extra spaces
    cleaned_transcription = " ".join(cleaned_transcription.split())

    return cleaned_transcription

# Combine transcriptions for each audio file
def combine_transcriptions(transcriptions_list: list) -> dict:
    combined_results = {}
    for audio_id in transcriptions_list[0].keys():
        # Step 1: Get all transcriptions for this audio file
        all_transcriptions = [t[audio_id] for t in transcriptions_list if audio_id in t]
        
        # Step 2: Remove hallucinations from each transcription
        unhallucinated_transcriptions = [remove_hallucinations_strict(t) for t in all_transcriptions]
        
        # Step 3: Select the longest valid transcription
        longest_transcription = max(unhallucinated_transcriptions, key=len, default="")

        # Store final cleaned result
        combined_results[audio_id] = longest_transcription

    return combined_results

# Save combined transcriptions to a file
def save_combined_transcriptions(combined_results: dict, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as file:
        for audio_id, transcription in combined_results.items():
            file.write(f"{audio_id} {transcription}\n")

if __name__ == "__main__":
    # Example input files
    file1 = "file2.txt"
    file2 = "file3.txt"
    file3 = "file4.txt"
    file4 = "file5.txt"
    file5 = "file1.txt"
    
    # Load transcriptions from each ASR model's output
    transcriptions1 = load_transcriptions(file1)
    transcriptions2 = load_transcriptions(file2)
    transcriptions3 = load_transcriptions(file3)
    transcriptions4 = load_transcriptions(file4)
    transcriptions5 = load_transcriptions(file5)    
    
    # Combine transcriptions
    combined_transcriptions = combine_transcriptions([transcriptions1, transcriptions2, transcriptions3, transcriptions4, transcriptions5])

    # Save to an output file
    output_file = "combined_transcriptions.txt"
    save_combined_transcriptions(combined_transcriptions, output_file)

    print(f"Combined transcriptions saved to {output_file}")
