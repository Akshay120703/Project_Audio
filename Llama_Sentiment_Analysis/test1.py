import subprocess
import json

def ollama_invoke(prompt, model="llama3.1"):
    command = f"ollama run {model}"
    try:
        result = subprocess.run(command, input=prompt, text=True, shell=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {result.stderr.strip()}")
        
        # Log raw output for debugging
        print("Raw Output:", result.stdout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Error invoking Ollama CLI: {e}")

if __name__ == "__main__":
    prompt = "Hello, how are you?"
    try:
        response = ollama_invoke(prompt)
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
