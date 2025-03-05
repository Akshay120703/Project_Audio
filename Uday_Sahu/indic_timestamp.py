from transformers import AutoProcessor, AutoModelForCTC
import torch
import torchaudio
import torchaudio.transforms as T

# Load the Indic Conformer Model and Processor
model_path = r"D:\whisper_hindi\model"  # Change this to your local model path
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCTC.from_pretrained(model_path)

# Load and preprocess the audio
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Ensure the audio is in the expected sample rate (16kHz)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform, target_sample_rate

audio_path = "sample_audio.wav"
waveform, sample_rate = load_audio(audio_path)

# Convert audio to input format
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# Forward pass through the model
with torch.no_grad():
    logits = model(**inputs).logits  # Shape: (Batch, Time, Vocab_size)

# Get predicted token IDs
predicted_ids = torch.argmax(logits, dim=-1)

# Convert token IDs to text and get offsets
decoded = processor.batch_decode(predicted_ids, output_offsets=True)
transcription = decoded["text"][0]
offsets = decoded["offsets"][0]  # Character-level offsets

# Compute frame duration
num_frames = logits.shape[1]  # Total time steps from model output
audio_duration = waveform.shape[1] / sample_rate  # Total audio duration in seconds
time_per_step = audio_duration / num_frames  # Time duration of each frame

# Extract word-level timestamps with confidence
word_timestamps = []
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Compute log probabilities
confidence_scores = torch.exp(log_probs)  # Convert log probs to confidence scores

prev_word = ""
word_start = None
word_confidence = []
for idx, (char, (start, end)) in enumerate(zip(decoded["text"][0], offsets)):
    if char == " ":
        if prev_word:  # If a word is complete
            word_end = end * time_per_step
            avg_confidence = sum(word_confidence) / len(word_confidence) if word_confidence else 1.0
            word_timestamps.append({
                "word": prev_word,
                "start_time": round(word_start, 3),
                "end_time": round(word_end, 3),
                "confidence": round(avg_confidence, 3),
            })
        prev_word = ""
        word_start = None
        word_confidence = []
    else:
        if not prev_word:
            word_start = start * time_per_step
        prev_word += char
        word_confidence.append(confidence_scores[0, start].item())

# Print word-level timestamps with confidence
for item in word_timestamps:
    print(item)
