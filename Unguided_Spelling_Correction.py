import subprocess
import pandas as pd
import re

# Function to invoke Ollama CLI for text enhancement
def ollama_text_enhance(prompt, model="llama3.1"):
    command = f"ollama run {model}"
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
    Only modify spelling errors and add **stopwords** when required to correct grammatical context (like में, का, की).  
    Do not add extra words, phrases, or change the meaning beyond spelling correction.  

    **Examples**:
    Predicted: "हिंदी मे बड़ी समस्या है।"
    Corrected: "हिंदी में बड़ी समस्या है।"

    Predicted: "संजय"
    Corrected: "संजे"

    Input Transcription: "{text}"
    Output: Correct and clean the transcription.
    """
    return ollama_text_enhance(prompt)

# Function to extract only the cleaned transcription
def extract_clean_transcription(enhanced_text):
    match = re.search(r'(?:"|\n)([^"\n]+)"?\s*$', enhanced_text.strip())
    return match.group(1) if match else enhanced_text

# Function to generate enhanced transcription log
def generate_log_file(df, log_file):
    with open(log_file, "w", encoding="utf-8") as log:
        for index, row in df.iterrows():
            transcription = row['Transcriptions']
            try:
                enhanced_text = enhance_transcription(transcription)
                clean_transcription = extract_clean_transcription(enhanced_text)
                log.write(f"Row {index} - Original: {transcription}\nEnhanced: {clean_transcription}\n\n")
                df.at[index, 'Enhanced Transcriptions'] = clean_transcription
            except Exception as e:
                log.write(f"Row {index} - Error enhancing transcription: {e}\n")
                df.at[index, 'Enhanced Transcriptions'] = transcription

# Function to save transcription output file
def save_output_file(df, output_file):
    df[['Audio Path', 'Enhanced Transcriptions']].to_csv(output_file, index=False, encoding="utf-8")
    print(f"Transcription output saved to {output_file}.")

# Main function to process transcription
def process_transcription_file(input_file, output_file, log_file):
    df = pd.read_csv(input_file)
    if 'Audio Path' not in df.columns or 'Transcriptions' not in df.columns:
        raise ValueError("Input file must contain 'Audio Path' and 'Transcriptions' columns.")

    df['Enhanced Transcriptions'] = ""
    generate_log_file(df, log_file)
    save_output_file(df, output_file)
    print(f"Log saved to {log_file}.")

# Define paths for input, output, and log files
input_file = "Sample1.txt"  # Replace with your input file
output_file = "transcriptions_output.txt"  # Output file
log_file = "transcription_log.txt"  # Log file

# Process the transcription file
process_transcription_file(input_file, output_file, log_file)
