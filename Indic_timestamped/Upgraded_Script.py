import torch
import numpy as np
import librosa
from nemo.collections.asr.models import EncDecCTCModel
from ctc_segmentation import CtcSegmentationParameters, prepare_text, ctc_segmentation

# ---------------- Configuration ----------------
audio_file = "/home/yolo/Desktop/Ensemble_testing/sample_audio/01-00281-01.mp3"
model_path = "/home/yolo/Desktop/Ensemble_testing/indic_conformer/IndicConformer/ai4b_indicConformer_hi.nemo"

reference_text = "your reference transcription here"

# ---------------- Load Model ----------------
model = EncDecCTCModel.restore_from(restore_path=model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------- Load & Preprocess Audio ----------------
audio, sr = librosa.load(audio_file, sr=16000)  # Load at 16kHz

# Convert to tensor (1, time) and move to GPU
input_tensor = torch.tensor(audio).float().unsqueeze(0).to(device)
length_tensor = torch.tensor([input_tensor.shape[1]]).to(device)

# Extract input features properly
with torch.no_grad():
    input_features = model.preprocessor(input_signal=input_tensor, length=length_tensor)
    processed_signal, processed_length = input_features  # Unpack tuple

# ---------------- Run Inference ----------------
with torch.no_grad():
    logits, log_probs_len = model.encoder(input_signal=processed_signal, length=processed_length)

# Convert logits to log probabilities (for CTC)
log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()[0]

# ---------------- Forced Alignment Setup ----------------
params = CtcSegmentationParameters()
params.index_duration = 0.02  # Each frame is ~20ms
params.blank = model.tokenizer.blank_id  # Get blank token ID

# Tokenize the reference text
tokenized_ref = model.tokenizer.text_to_ids(reference_text)
ground_truth_mat = prepare_text(params, [" ".join(map(str, tokenized_ref))])

# Run forced alignment
segmentation_result = ctc_segmentation(params, log_probs, ground_truth_mat)
segmentation = segmentation_result[0]  # Extract segmented tokens

# ---------------- Group Tokens into Words ----------------
words = reference_text.split()
word_level_info = []
token_index = 0

try:
    for word in words:
        token_ids = model.tokenizer.text_to_ids(word)
        num_tokens = len(token_ids)

        if token_index + num_tokens > len(segmentation):
            print(f"Warning: Not enough token segments for '{word}'.")
            break

        token_segments = segmentation[token_index: token_index + num_tokens]
        start_time = token_segments[0]['start']
        end_time = token_segments[-1]['end']
        confidence = np.mean([seg.get('score', 1.0) for seg in token_segments])

        word_level_info.append({
            "word": word,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence
        })

        token_index += num_tokens

except Exception as e:
    print("Error during token grouping:", e)

# ---------------- Output Word-Level Alignment ----------------
print("\nWord-level alignment:")
for info in word_level_info:
    print(info)
