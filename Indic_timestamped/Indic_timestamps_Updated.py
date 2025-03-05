import torch
import numpy as np
import librosa
from nemo.collections.asr.models import EncDecCTCModel
from ctc_segmentation import CtcSegmentationParameters, prepare_text, ctc_segmentation

# ---------------- Configuration ----------------
# Path to the audio file and Nemo model
audio_file = "/home/yolo/Desktop/Ensemble_testing/sample_audio/01-00281-01.mp3"
model_path = "/home/yolo/Desktop/Ensemble_testing/indic_conformer/IndicConformer/ai4b_indicConformer_hi.nemo"

# Replace with your actual reference transcription.
reference_text = "your reference transcription here"

# ---------------- Load Model ----------------
model = EncDecCTCModel.restore_from(restore_path=model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------- Load and Preprocess Audio ----------------
# Load audio (resample to 16 kHz)
audio, sr = librosa.load(audio_file, sr=16000)

# Convert to tensor and add batch dimension: shape (1, time)
input_tensor = torch.tensor(audio).unsqueeze(0).to(device)
length_tensor = torch.tensor([input_tensor.shape[1]]).to(device)

# Obtain input features using the model’s preprocessor
input_features = model.preprocessor(input_signal=input_tensor, length=length_tensor)

# ---------------- Run Inference ----------------
# Get raw logits from the model; these are not normalized
with torch.no_grad():
    logits = model(input_signal=input_features)
# Convert logits to log probabilities and remove batch dimension
log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()[0]

# ---------------- Forced Alignment Setup ----------------
params = CtcSegmentationParameters()
# Set the duration for each index (frame); adjust as needed.
params.index_duration = 0.02  
# Set the blank token id (as defined by your model’s tokenizer)
params.blank = model.tokenizer.blank_id

# Prepare the ground truth transcript for the forced aligner.
ground_truth_mat = prepare_text(params, [reference_text])

# Run forced alignment to obtain token-level segmentation.
segmentation_result = ctc_segmentation(params, log_probs, ground_truth_mat)
segmentation = segmentation_result[0]

# (Optional) Print segmentation output for inspection.
print("Segmentation output:")
print(segmentation)

# ---------------- Group Tokens into Words ----------------
words = reference_text.split()
word_level_info = []
token_index = 0

try:
    for word in words:
        # Convert the word into token IDs using the model's tokenizer.
        token_ids = model.tokenizer.text_to_ids(word)
        num_tokens = len(token_ids)
        
        # Ensure there are enough token segments; if not, warn and break.
        if token_index + num_tokens > len(segmentation):
            print(f"Warning: Not enough token segments for word '{word}'. Skipping remaining words.")
            break
        
        # Get the segmentation info for tokens corresponding to this word.
        token_segments = segmentation[token_index: token_index + num_tokens]
        # Use the start time of the first token and end time of the last token.
        start_time = token_segments[0]['start']
        end_time = token_segments[-1]['end']
        # Calculate average confidence (or score) across the tokens.
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
