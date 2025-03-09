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
processor = Wav2Vec2Processor.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec").to("cpu")

# ========== Paths ==========
audio_folder = "/home/pc/Desktop/audio/dir"
output_file = "/home/pc/Desktop/audio/output_2/output.txt"

# ========== Function to Load Audio ==========
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    print(f"Audio loaded. Length: {len(waveform)} samples, Sample rate: {sr} Hz.")
    return waveform

# ========== Function to Extract Wav2Vec Features ==========
def extract_wav2vec_features(audio):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cpu")
    print(f"Feature extraction input shape: {input_values.shape}")  # Debugging shape
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print(f"Feature extraction complete. Shape of probabilities: {probs.shape}.")  # Debugging shape
    return probs.cpu().numpy()

# ========== Function to Detect Blank Segments ==========
def detect_blank_segments(probs, threshold=0.5):
    blank_segments = []
    frame_duration = 0.02  # 20ms per frame

    num_frames = probs.shape[1]  # Total number of frames
    print(f"Number of frames: {num_frames}")
    
    blank_start = None  # Start of a blank segment

    for frame_idx in range(num_frames):
        frame_max_prob = np.max(probs[0, frame_idx])  # Max prob of this frame
        print(f"Frame {frame_idx}: Max prob = {frame_max_prob}")

        # Detect if the frame is blank based on the probability threshold
        if frame_max_prob < threshold:
            if blank_start is None:
                blank_start = frame_idx  # Start of blank segment
        else:
            if blank_start is not None:
                blank_end = frame_idx  # End of blank segment
                blank_segments.append((blank_start * frame_duration, blank_end * frame_duration))
                blank_start = None  # Reset for the next blank segment

    # If the last frame is part of a blank segment
    if blank_start is not None:
        blank_segments.append((blank_start * frame_duration, num_frames * frame_duration))

    return blank_segments

# ========== Function to Align Words Using DTW ==========
def dtw_alignment(ctc_words, wav2vec_probs):
    num_frames = len(wav2vec_probs)
    frame_duration = 0.02  # 20ms per frame
    
    word_timestamps = []
    frame_idx_start = 0
    
    # Iterate over words in CTC transcription
    for word in ctc_words:
        # Find the word's corresponding frames by examining Wav2Vec2's probabilities
        word_length_in_frames = int(num_frames / len(ctc_words))  # Simplified for now
        frame_idx_end = min(frame_idx_start + word_length_in_frames, num_frames)
        
        word_frames = wav2vec_probs[frame_idx_start:frame_idx_end]
        
        # Check if the frames are empty or invalid
        if len(word_frames) == 0 or np.any(np.isnan(word_frames)):
            print(f"Skipping word: {word} due to invalid frames. Word length: {len(word_frames)}")
            frame_idx_start = frame_idx_end
            continue
        
        # Log the word and frames for debugging
        print(f"Aligning word: '{word}' from frame {frame_idx_start} to {frame_idx_end}")
        
        # Calculate the start and end times (in seconds)
        start_time = frame_idx_start * frame_duration
        end_time = frame_idx_end * frame_duration
        
        # Calculate average confidence score across frames for this word
        confidence_score = np.mean(np.max(word_frames, axis=-1))  # max probability per frame
        
        # Append the word's start time, end time, and confidence score
        word_timestamps.append((word, start_time, end_time, confidence_score))
        
        # Update the frame index for the next word
        frame_idx_start = frame_idx_end

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

            # Log number of frames and shape of probabilities for debugging
            print(f"Number of frames: {len(wav2vec_probs)}")
            print(f"Shape of probabilities: {wav2vec_probs.shape}")
            
            # Get blank segments
            blank_segments = detect_blank_segments(wav2vec_probs)

            # Write results
            f.write(f"{filename}:\n")
            f.write(f"Blank Segments: {blank_segments}\n")  # Write blank segments

            # Get word-level timestamps and confidence scores
            word_timestamps = dtw_alignment(ctc_words, wav2vec_probs)

            for word, start_time, end_time, confidence in word_timestamps:
                f.write(f"{word}: {start_time:.2f} sec - {end_time:.2f} sec - {confidence:.2f} confidence\n")
            f.write("\n")

            print(f"Processed {filename}: {len(ctc_words)} words aligned.")

print(f"Transcriptions with timestamps and blank segments saved to {output_file}")

