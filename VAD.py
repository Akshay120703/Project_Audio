import webrtcvad
from pydub import AudioSegment
import numpy as np

def vad_denoising(audio_path, output_path, aggressiveness=2):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    samples = np.array(audio.get_array_of_samples())
    
    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # Frame parameters
    frame_duration = 30  # ms
    sample_rate = audio.frame_rate
    frame_size = int(sample_rate * frame_duration / 1000)
    
    denoised_samples = []
    
    # Split into frames and apply VAD
    for start in range(0, len(samples), frame_size):
        frame = samples[start:start + frame_size]
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if is_speech:
            denoised_samples.extend(frame)
        else:
            denoised_samples.extend(np.zeros_like(frame))  # Optional: add silence instead of noise
    
    # Create and save the denoised audio
    denoised_audio = audio._spawn(denoised_samples)
    denoised_audio.export(output_path, format="wav")
    print(f"Denoised audio saved at {output_path}")

# Example usage:
for filename in os.listdir(input_folder):
    if filename.endswith(".mp3"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_vad_denoised.wav")
        vad_denoising(input_path, output_path)
