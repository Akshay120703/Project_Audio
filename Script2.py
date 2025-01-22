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

def translate_gurmukhi_to_hindi(input_file, output_file):
    """
    Translates text from a Gurmukhi (Punjabi) text file to Hindi using the Ollama CLI and llama3.1 model.

    :param input_file: Path to the input text file containing Gurmukhi script.
    :param output_file: Path to the output text file to save Hindi script.
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        return

    # Read the content of the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        gurmukhi_text = file.read()

    # Check if the text is empty
    if not gurmukhi_text.strip():
        print("Error: The input file is empty.")
        return

    # Construct the prompt for the translation task
    prompt = (
        "Translate the following text from Gurmukhi script (Punjabi) to Hindi script: \n"
        f"{gurmukhi_text}"
    )

    # Invoke the Ollama CLI to get the translation
    try:
        hindi_text = ollama_text_enhance(prompt, model="llama3.1")
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    # Write the Hindi text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(hindi_text)

    print(f"Translation complete. The Hindi text has been saved to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    input_path = "gurmukhi_input.txt"  # Path to the input file
    output_path = "hindi_output.txt"   # Path to the output file

    translate_gurmukhi_to_hindi(input_path, output_path) 
