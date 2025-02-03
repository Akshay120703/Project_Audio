import subprocess
import json
import re
from collections import Counter

# Function to invoke the Ollama CLI
def ollama_invoke(prompt, model="llama3.1"):
    """Invokes Llama 3.1 using Ollama CLI and returns the processed output."""
    command = f"ollama run {model}"
    try:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            shell=True,
            capture_output=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {result.stderr.strip()}")

        return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Error invoking Ollama CLI: {e}")

# Step 1: Read ASR Outputs
def read_transcripts(filepaths):
    """Reads multiple transcript files and returns a dictionary of transcripts mapped to audio file paths."""
    transcripts = {}

    for path in filepaths:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    audio_file, text = parts
                    if audio_file not in transcripts:
                        transcripts[audio_file] = []
                    transcripts[audio_file].append(text.split())  # Store tokenized transcript

    return transcripts

# Step 2: Remove Hallucinations
def remove_hallucinations(transcripts):
    """Removes hallucinated outputs where a 1-5 word sequence is repeated more than 5 times."""
    filtered_transcripts = {}

    for audio_file, transcript_list in transcripts.items():
        cleaned_list = []
        for transcript in transcript_list:
            word_counts = Counter(transcript)
            hallucinated_words = {word for word, count in word_counts.items() if count > 5}
            cleaned_list.append([word for word in transcript if word not in hallucinated_words])
        filtered_transcripts[audio_file] = cleaned_list

    return filtered_transcripts

# Step 3: Align Words in Correct Order
def align_words(transcripts):
    """Aligns words from different ASR models while maintaining the original order."""
    aligned_transcripts = {}

    for audio_file, transcript_list in transcripts.items():
        combined_sequence = []
        seen_words = set()

        # Align words while keeping order intact
        for transcript in transcript_list:
            temp_seq = []
            for word in transcript:
                if word not in seen_words:
                    temp_seq.append(word)
                    seen_words.add(word)
            combined_sequence.extend(temp_seq)

        aligned_transcripts[audio_file] = combined_sequence

    return aligned_transcripts

# Step 4: Use Llama to Detect and Remove Incorrect Spellings
def correct_spelling_with_llama(text):
    """Uses Llama 3.1 to detect and correct spelling variations."""
    prompt = f"""
    The following Hindi transcription may contain the same words with different spellings appearing within a 5-word range.
    Detect incorrectly spelled words and return only the corrected sentence.

    Input: "{text}"
    Output (corrected sentence):
    """
    response = ollama_invoke(prompt)
    return response

# Step 5: Majority Voting for Final Cleanup
def majority_voting_cleanup(transcripts):
    """Applies majority voting to remove incorrect words based on frequency."""
    final_transcripts = {}

    for audio_file, words in transcripts.items():
        word_counts = Counter(words)
        cleaned_words = [word for word in words if word_counts[word] > 1]  # Keep words appearing more than once
        final_transcripts[audio_file] = cleaned_words

    return final_transcripts

# Step 6: Save the Final Output
def save_output(transcripts, output_path):
    """Saves the ensembled transcripts to a file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for audio_file, text in transcripts.items():
            f.write(f"{audio_file}, {' '.join(text)}\n")

# Main Function
def ensemble_asr_outputs(filepaths, output_path):
    """Performs ASR ensembling by following all defined steps."""
    transcripts = read_transcripts(filepaths)
    transcripts = remove_hallucinations(transcripts)
    transcripts = align_words(transcripts)

    # Apply Llama 3.1 for spelling correction
    corrected_transcripts = {}
    for audio_file, text in transcripts.items():
        corrected_transcripts[audio_file] = correct_spelling_with_llama(" ".join(text)).split()

    # Apply Majority Voting
    transcripts = majority_voting_cleanup(corrected_transcripts)

    # Save final output
    save_output(transcripts, output_path)
    print(f"Ensembled transcript saved to {output_path}")

# Example usage
filepaths = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]
output_file = "ensembled_output.txt"
ensemble_asr_outputs(filepaths, output_file)
