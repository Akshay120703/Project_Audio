import os
import subprocess

def ollama_text_enhance(prompt, model="llama3.1"):
    command = f"ollama run {model}"
    try:
        result = subprocess.run(command, input=prompt, text=True, shell=True, capture_output=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {result.stderr.strip()}")
        return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Error invoking Ollama CLI: {e}")

def translate_urdu_to_hindi(input_file, output_file):
    """
    Translates Urdu text to Hindi using Ollama and the llama3.1 model.

    Args:
        input_file (str): Path to the input text file containing Urdu script.
        output_file (str): Path to the output text file for the Hindi script.
    """
    # Ensure input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

    # Read the Urdu text from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        urdu_text = file.read()

    # Prepare prompt for Ollama
    prompt = f"Translate the following Urdu text to Hindi:\n{urdu_text}"

    # Use Ollama CLI for translation
    try:
        hindi_text = ollama_text_enhance(prompt, model="llama3.1")
    except RuntimeError as e:
        raise RuntimeError(f"Translation failed: {e}")

    # Write the Hindi text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(hindi_text)

    print(f"Translation completed successfully. Output written to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    input_path = "urdu_input.txt"  # Replace with your input file path
    output_path = "hindi_output.txt"  # Replace with your desired output file path
    try:
        translate_urdu_to_hindi(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}") 
