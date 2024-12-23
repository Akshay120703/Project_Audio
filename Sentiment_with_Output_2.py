from langchain_ollama import OllamaLLM
import pandas as pd
import json
import re

# Initialize the Llama model using OllamaLLM
model = OllamaLLM(model="llama3.1")

# Define the function to analyze transcription
def analyze_transcription(text):
    """Process the transcription using the model."""
    # Refined prompt with examples and clearer instructions for better output
    prompt = f"""
    Analyze the given Hindi transcription for sentiment and provide a structured JSON response.

    **Ensure accurate classifications with examples**:
    - "मैं खुश हूँ।" → Positive, Confidence: 1.0
    - "मैं तुम्हें मार दूंगा।" → Negative, dangerous, Confidence: 1.0
    - "आज का मौसम अच्छा है।" → Neutral, Confidence: 0.8

    Provide your output in this strict JSON format:
    BEGIN JSON
    {{
        "Sentiment Classification": "Positive/Negative/Neutral",
        "Sentiment Category": "dangerous/threatening/toxic/abnormal (if applicable, otherwise empty)",
        "Confidence Score": 0.0-1.0,
        "Explanation": "Explanation in Hindi, 50 words or less.",
        "NER": ["List of key named entities"],
        "Keywords": ["Critical words detected"]
    }}
    END JSON

    Transcription: "{text}"
    """
    # Pass the prompt as a string to the model
    result = model.invoke(prompt)
    return result

# Define the function to process TXT file
def process_txt(input_txt, output_txt, log_file):
    """Process the input TXT and save results to the output TXT."""
    # Load the input TXT as a DataFrame
    df = pd.read_csv(input_txt, sep=",")  # Comma-separated format

    # Set the "transcript_id" as the index (string-based)
    df.set_index("transcript_id", inplace=True)

    # Add new columns for results
    df["Sentiment Classification"] = ""
    df["Sentiment Category"] = ""
    df["Confidence Score"] = ""
    df["Explanation"] = ""
    df["NER"] = ""
    df["Keywords"] = ""

    with open(log_file, "w", encoding="utf-8") as log:
        # Process each row using string-based index
        for transcript_id, row in df.iterrows():
            text = row["transcript"]
            try:
                # Generate analysis
                result = analyze_transcription(text)

                # Log raw result for debugging
                log.write(f"Transcription ID: {transcript_id}\n")
                log.write(f"Raw Model Output:\n{result}\n\n")

                # Extract JSON from the response
                json_match = re.search(r"BEGIN JSON(.*?)END JSON", result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    parsed = json.loads(json_str)  # Parse JSON output

                    # Update the DataFrame with parsed results
                    df.at[transcript_id, "Sentiment Classification"] = parsed.get("Sentiment Classification", "")
                    df.at[transcript_id, "Sentiment Category"] = parsed.get("Sentiment Category", "")
                    df.at[transcript_id, "Confidence Score"] = parsed.get("Confidence Score", "")
                    df.at[transcript_id, "Explanation"] = parsed.get("Explanation", "")
                    df.at[transcript_id, "NER"] = ", ".join(parsed.get("NER", []))
                    df.at[transcript_id, "Keywords"] = ", ".join(parsed.get("Keywords", []))
                else:
                    # Log an error if JSON markers are not found
                    log.write(f"Error: JSON markers not found for ID {transcript_id}. Assigning default values.\n")
                    df.at[transcript_id, "Sentiment Classification"] = "Neutral"
                    df.at[transcript_id, "Sentiment Category"] = ""
                    df.at[transcript_id, "Confidence Score"] = 0.5
                    df.at[transcript_id, "Explanation"] = "Model output did not contain JSON markers."
                    df.at[transcript_id, "NER"] = ""
                    df.at[transcript_id, "Keywords"] = ""

            except Exception as e:
                # Log the exception
                log.write(f"Error processing transcription ID {transcript_id}: {e}\n")
                log.write(f"Transcription causing the error: {text}\n")
                df.at[transcript_id, "Explanation"] = f"Error: {str(e)}"

    # Save results to the output TXT (comma-separated format)
    df.to_csv(output_txt, sep=",", encoding="utf-8")
    print(f"Analysis complete. Results saved to {output_txt}. Log saved to {log_file}.")

# Define input, output, and log file paths
input_txt = "input.txt"  # Input TXT file path
output_txt = "output.txt"  # Output TXT file path
log_file = "processing_log.txt"  # Log file path

# Run the TXT processing function
process_txt(input_txt, output_txt, log_file)
