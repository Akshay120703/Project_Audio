import os
import torch
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
import torchaudio
import torchaudio.transforms as transforms
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ========== Load IndicConformer Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conformer_model = EncDecCTCModel.restore_from(restore_path="/home/pc/Desktop/IndicConformer/ai4b_indicConformer_hi.nemo")
conformer_model.freeze()
conformer_model = conformer_model.to(device)
conformer_model.cur_decoder = "ctc"  # Set to CTC decoding

# ========== Load Wav2Vec2 Model for Alignment ==========
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53").to("cuda")

# ========== Paths ==========
audio_folder = "/home/pc/Desktop/audio/dir"
output_file = "/home/pc/Desktop/IndicConformer/predicted_transcript_with_timestamps.txt"

# ========== Function to Load Audio ==========
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform

# ========== Function to Extract Wav2Vec Features ==========
def extract_wav2vec_features(audio):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.cpu().numpy()

# ========== Function to Align Words Using DTW ==========
def dtw_alignment(ctc_words, wav2vec_probs):
    ctc_timings = np.linspace(0, len(wav2vec_probs), len(ctc_words) + 1)
    wav2vec_timings = np.arange(len(wav2vec_probs))
    
    # Compute cost matrix
    cost_matrix = cdist(ctc_timings.reshape(-1, 1), wav2vec_timings.reshape(-1, 1))
    
    # Solve alignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    word_timestamps = []
    for word_idx, frame_idx in zip(row_ind, col_ind):
        if word_idx < len(ctc_words):  # Ignore last index as it's a buffer
            start_time = frame_idx * (1 / 50)  # Assuming 50fps from Wav2Vec
            confidence_score = np.mean(wav2vec_probs[frame_idx])  # Averaged confidence score
            word_timestamps.append((ctc_words[word_idx], start_time, confidence_score))
    
    return word_timestamps

# ========== Main Processing ==========
with open(output_file, "w") as f:
    for filename in os.listdir(audio_folder):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(audio_folder, filename)
            
            # Transcribe using IndicConformer
            ctc_text = conformer_model.transcribe([file_path], batch_size=1)[0]
            if isinstance(ctc_text, list):
                ctc_text = ctc_text[0]
            ctc_words = ctc_text.split()

            # Load audio
            audio = load_audio(file_path)

            # Extract Wav2Vec features
            wav2vec_probs = extract_wav2vec_features(audio)

            # Get word-level timestamps and confidence scores
            word_timestamps = dtw_alignment(ctc_words, wav2vec_probs)

            # Write results
            f.write(f"{filename}:\n")
            for word, timestamp, confidence in word_timestamps:
                f.write(f"{word}: {timestamp:.2f} sec, Confidence: {confidence:.2f}\n")
            f.write("\n")

            print(f"Processed {filename}: {len(ctc_words)} words aligned.")

print(f"Transcriptions with timestamps saved to {output_file}")
