import os
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ========== Load IndicConformer Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conformer_model = EncDecCTCModel.restore_from(restore_path="/home/pc/Desktop/IndicConformer/ai4b_indicConformer_hi.nemo")
conformer_model.freeze()
conformer_model = conformer_model.to(device)

# ========== Load Wav2Vec2 Model for Alignment ==========
processor = Wav2Vec2Processor.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec").to("cpu")

# ========== Paths ==========
audio_folder = "/home/pc/Desktop/audio/dir"
output_file = "/home/pc/Desktop/audio/output_2/output.txt"

# ========== Load Audio ==========
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform

# ========== Extract Frame-Level Probabilities ==========
def extract_wav2vec_features(audio):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cpu")
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.cpu().numpy()

# ========== Word-Level Timestamping with Binary Search ==========
def get_word_timestamps(text, probs, frame_rate=50):  
    """
    Perform binary search-based timestamping.
    - frame_rate: 50 fps (20ms per frame, assuming 16kHz audio)
    """
    words = text.split()
    num_frames = probs.shape[1]  
    frame_duration = 1 / frame_rate  # 20ms per frame → 0.02 sec

    word_timestamps = []
    start_frame = 0

    for word in words:
        # Start binary search from (start_frame, start_frame + 0.5 sec in frames)
        low, high = start_frame, min(start_frame + int(0.5 / frame_duration), num_frames - 1)

        while low < high:
            mid = (low + high) // 2  # Middle frame for binary search
            
            # If next frame is space and next-to-next is not space, it's a word boundary
            if mid < num_frames - 2 and probs[0, mid + 1].argmax() == 0 and probs[0, mid + 2].argmax() != 0:
                high = mid  # Narrow search to find exact word-end
            else:
                low = mid + 1

        # Word timestamp confirmation
        end_frame = low
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        word_confidence = np.mean(np.max(probs[0, start_frame:end_frame], axis=-1))

        # Append timestamps
        word_timestamps.append((word, start_time, end_time, word_confidence))
        start_frame = end_frame  # Move to the next word's start

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

            # Load audio
            audio = load_audio(file_path)

            # Extract Wav2Vec features
            wav2vec_probs = extract_wav2vec_features(audio)

            # Get word-level timestamps
            word_timestamps = get_word_timestamps(ctc_text, wav2vec_probs)

            # Write results
            f.write(f"{filename}:\n")
            for word, start_time, end_time, confidence in word_timestamps:
                f.write(f"{word}: {start_time:.2f} sec - {end_time:.2f} sec - {confidence:.2f} confidence\n")
            f.write("\n")

            print(f"Processed {filename}: {len(word_timestamps)} words aligned.")

print(f"Word timestamps saved to {output_file}")
