import subprocess
import json

# Function to invoke the Ollama CLI
def ollama_invoke(prompt, context, model="llama3.1"):
    """
    Invoke the Ollama CLI with a given prompt and context.

    Args:
        prompt (str): The user prompt to send to the model.
        context (list): The conversation history for context.
        model (str): The model name to use.

    Returns:
        str: The model's response.
    """
    # Combine context and prompt
    full_prompt = "\n".join(context + [prompt])
    command = f"ollama run {model}"
    try:
        result = subprocess.run(
            command,
            input=full_prompt,
            text=True,
            shell=True,
            capture_output=True,
            encoding="utf-8"  # Explicitly specify UTF-8 encoding
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ollama CLI error: {result.stderr.strip()}")
        
        # Commented out raw output logging
        # print("Raw Output:", result.stdout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    except Exception as e:
        raise RuntimeError(f"Error invoking Ollama CLI: {e}")

if __name__ == "__main__":
    context = []  # Initialize context as an empty list
    model_name = "llama3.1"  # Define the model name
    
    print("Welcome to the interactive chat with context! Type 'exit' to quit.\n")
    while True:
        prompt = input("You: ")  # Get user input
        if prompt.lower() == "exit":  # Exit condition
            print("Goodbye!")
            break
        try:
            # Generate response
            response = ollama_invoke(prompt, context, model=model_name)
            print(f"Model: {response}")
            
            # Update context with the latest prompt and response
            context.append(f"You: {prompt}")
            context.append(f"Model: {response}")
        except Exception as e:
            print(f"Error: {e}")
