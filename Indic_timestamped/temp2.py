import torch
import numpy as np
import librosa
import json
from nemo.collections.asr.models import EncDecCTCModel

# Ensure ctc_segmentation is installed.
try:
    from ctc_segmentation import CtcSegmentationParameters, prepare_text, ctc_segmentation
except ImportError as e:
    raise ImportError("ctc_segmentation module not found. Please install it with 'pip install ctc_segmentation'") from e

# --------------------- Configuration ---------------------
# Path to your audio file and Nemo model file.
audio_file = "/home/yolo/Desktop/Ensemble_testing/sample_audio/01-00281-01.mp3"
model_path = "/home/yolo/Desktop/Ensemble_testing/indic_conformer/IndicConformer/ai4b_indicConformer_hi.nemo"

# IMPORTANT: Provide the correct reference transcription that exactly matches the speech in the audio.
reference_text = "your reference transcription here"

# --------------------- Load Model ---------------------
model = EncDecCTCModel.restore_from(restore_path=model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --------------------- Load and Preprocess Audio ---------------------
# Load the audio file (resampling to 16 kHz).
audio, sr = librosa.load(audio_file, sr=16000)
# Convert audio to a tensor and add a batch dimension.
input_tensor = torch.tensor(audio).unsqueeze(0).to(device)
# Create a tensor representing the length of the audio.
length_tensor = torch.tensor([input_tensor.shape[1]]).to(device)

# Process the raw audio into input features required by the model.
input_features = model.preprocessor(input_signal=input_tensor, length=length_tensor)

# --------------------- Model Inference ---------------------
# Obtain the raw logits from the model.
with torch.no_grad():
    logits = model(input_signal=input_features)
# Move logits to CPU.
logits = logits.cpu()
# Convert logits to log probabilities.
log_probs = torch.nn.functional.log_softmax(logits, dim=-1).numpy()[0]  # shape: (T, vocab_size)

# --------------------- Forced Alignment Setup ---------------------
params = CtcSegmentationParameters()
# Set the duration (in seconds) of each frame/index (adjust if necessary).
params.index_duration = 0.02  
# Set the blank token ID from the model's tokenizer.
params.blank = model.tokenizer.blank_id

# Prepare the reference transcription for alignment.
ground_truth_mat = prepare_text(params, [reference_text])

# Run forced alignment to obtain token-level segmentation.
segmentation_result = ctc_segmentation(params, log_probs, ground_truth_mat)
segmentation = segmentation_result[0]

print("Token-level segmentation output:")
print(segmentation)

# --------------------- Group Tokens into Words ---------------------
words = reference_text.split()
word_level_info = []
token_index = 0

for word in words:
    # Convert the word to a list of token IDs.
    try:
        token_ids = model.tokenizer.text_to_ids(word)
    except Exception as e:
        print(f"Error converting word '{word}' to token IDs: {e}")
        continue
    num_tokens = len(token_ids)
    if num_tokens == 0:
        continue

    # Check if there are enough tokens in the segmentation result.
    if token_index + num_tokens > len(segmentation):
        print(f"Warning: Not enough token segments for word '{word}'. Stopping grouping.")
        break

    # Group the segmentation info corresponding to the tokens of this word.
    token_segments = segmentation[token_index: token_index + num_tokens]
    # Use the start time of the first token and the end time of the last token.
    start_time = token_segments[0].get('start', 0.0)
    end_time = token_segments[-1].get('end', 0.0)
    # Compute average confidence (or score) over the tokens.
    confidence = np.mean([seg.get('score', 1.0) for seg in token_segments])
    
    word_level_info.append({
        "word": word,
        "start_time": start_time,
        "end_time": end_time,
        "confidence": confidence
    })
    token_index += num_tokens

# --------------------- Output the Word-Level Alignment ---------------------
print("\nWord-level alignment info:")
print(json.dumps(word_level_info, indent=4, ensure_ascii=False))
