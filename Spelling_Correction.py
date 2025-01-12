import subprocess
import pandas as pd
import torch

# Function to invoke Ollama CLI for text enhancement
def ollama_text_enhance(prompt, model="llama3.1"):
    command = f"ollama run {model}"  # Removed --device flag
    try:
        result = subprocess.run(command, input=prompt, text=True, shell=True, capture_output=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {result.stderr.strip()}")
        return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Error invoking Ollama CLI: {e}")

# Function for spelling correction and text improvement
def enhance_transcription(text):
    prompt = f"""
    Please improve the given Hindi transcription for spelling, phonetic similarity, and semantic correctness.
    Normalize, filter trash words, and apply necessary word corrections.

    **Examples**:
    Predicted: "हिंदी मे बड़ी समस्या है।"
    Corrected: "हिंदी में बड़ी समस्या है।"

    Predicted: "संजय"
    Corrected: "संजे"

    Input Transcription: "{text}"
    Output: Correct and clean the transcription.
    """
    return ollama_text_enhance(prompt)

# Function to process transcription file and apply corrections
def process_transcription_file(input_file, output_file, log_file):
    df = pd.read_csv(input_file)
    if 'Audio Path' not in df.columns or 'Transcriptions' not in df.columns:
        raise ValueError("Input file must contain 'Audio Path' and 'Transcriptions' columns.")

    df['Enhanced Transcriptions'] = ""

    with open(log_file, "w", encoding="utf-8") as log:
        for index, row in df.iterrows():
            transcription = row['Transcriptions']
            try:
                enhanced_text = enhance_transcription(transcription)
                log.write(f"Row {index} - Original: {transcription}\nEnhanced: {enhanced_text}\n\n")
                df.at[index, 'Enhanced Transcriptions'] = enhanced_text
            except Exception as e:
                log.write(f"Row {index} - Error enhancing transcription: {e}\n")
                df.at[index, 'Enhanced Transcriptions'] = transcription

    df[['Audio Path', 'Enhanced Transcriptions']].to_csv(output_file, index=False, encoding="utf-8")
    print(f"Enhancement complete. Results saved to {output_file}. Log saved to {log_file}.")

# Define paths for input, output, and log files
input_file = "Sample1.txt"  # Replace with your input file
output_file = "transcriptions_output.txt"  # Output file
log_file = "transcription_log.txt"  # Log file

# Process the transcription file
process_transcription_file(input_file, output_file, log_file)
