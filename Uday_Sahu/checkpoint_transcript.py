import torch
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio

def load_model(checkpoint_path: str):
    """Loads the fine-tuned Whisper model from the checkpoint."""
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    processor = WhisperProcessor.from_pretrained(checkpoint_path)
    model.eval()
    return model, processor

def preprocess_audio(audio_path: str, processor):
    """Loads and preprocesses the audio file."""
    speech_array, sampling_rate = torchaudio.load(audio_path)
    if sampling_rate != 16000:
        speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
    inputs = processor(speech_array.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
    return inputs.input_features

def generate_transcript(model, processor, audio_path: str):
    """Generates a transcript from an audio file using the fine-tuned Whisper model."""
    input_features = preprocess_audio(audio_path, processor)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript

def process_audio_directory(model, processor, audio_dir: str, output_file: str):
    """Processes all audio files in a directory and saves transcripts."""
    transcripts = {}
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(audio_dir, file_name)
            transcript = generate_transcript(model, processor, audio_path)
            transcripts[file_name] = transcript
    
    with open(output_file, "w", encoding="utf-8") as f:
        for file_name, transcript in transcripts.items():
            f.write(f"{file_name}: {transcript}\n")
    
    print(f"Transcripts saved to {output_file}")

if __name__ == "__main__":
    checkpoint_path = "path/to/your/finetuned/checkpoint"  # Update this with your actual checkpoint path
    audio_dir = "path/to/your/audio/directory"  # Update this with your audio directory
    output_file = "transcripts.txt"
    
    model, processor = load_model(checkpoint_path)
    process_audio_directory(model, processor, audio_dir, output_file)
