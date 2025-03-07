import os
import torch
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
import numpy as np
import librosa

# Specify the path to your audio files and the output text file
audio_folder = '/home/pc/Desktop/audio/dir'
output_file = '/home/pc/Desktop/audio/output/output.txt'

# Initialize the device and the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncDecCTCModel.restore_from(restore_path='/home/pc/Desktop/IndicConformer/ai4b_indicConformer_hi.nemo')
model.freeze()
model = model.to(device)

# Set the decoder to CTC
model.cur_decoder = 'ctc'

# Function to get word-level timestamps and confidence scores
def get_word_level_timestamps_and_confidence(logits, threshold=0.5, hop_length=512, sample_rate=16000):
    """
    Extract word-level timestamps and confidence scores from CTC logits.
    Args:
        logits: Tensor of shape [time_steps, vocab_size]
        threshold: Probability threshold for determining token presence.
        hop_length: Number of samples between successive frames.
        sample_rate: The sample rate of the audio.

    Returns:
        word_times (list of tuples): Each tuple contains (word, start_time, end_time, confidence)
    """
    # Get token probabilities (softmax over logits to get probabilities)
    token_probabilities = logits.softmax(dim=-1)  # Shape: [time_steps, vocab_size]
    
    word_times = []
    word_confidences = []

    current_word_start = None
    current_word_end = None
    word_tokens = []

    for time_step in range(token_probabilities.shape[0]):
        # Get most likely token at each time step
        token_id = torch.argmax(token_probabilities[time_step]).item()

        if token_id == 0:  # Usually, 0 is the padding token in many tokenizers (adjust accordingly)
            continue

        word_tokens.append(token_id)

        if current_word_start is None:
            current_word_start = time_step
        current_word_end = time_step

        # Calculate confidence score for this word
        word_confidence = token_probabilities[time_step, token_id].item()
        
        if time_step == token_probabilities.shape[0] - 1 or token_id != torch.argmax(token_probabilities[time_step + 1]).item():
            # End of word reached, add the word and timestamp
            word_times.append((word_tokens, current_word_start, current_word_end))
            word_confidences.append(np.mean([token_probabilities[t, token_id].item() for t in range(current_word_start, current_word_end + 1)]))

            # Reset for next word
            current_word_start = None
            current_word_end = None
            word_tokens = []

    # Convert frame indices to time (in seconds)
    frame_duration = 1 / sample_rate * hop_length
    word_times_in_seconds = [(start * frame_duration, end * frame_duration) for _, start, end in word_times]
    
    # Combine word_times with their corresponding confidences
    result = [{
        "word": ' '.join(map(str, word_tokens)),
        "start_time": start,
        "end_time": end,
        "confidence": confidence
    } for (word_tokens, start, end), confidence in zip(word_times, word_confidences)]

    return result

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate through all audio files in the folder
    for filename in os.listdir(audio_folder):
        # Filter out only audio files (e.g., mp3, wav)
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            file_path = os.path.join(audio_folder, filename)
            
            # Load and preprocess the audio file into mel spectrogram
            try:
                audio_input, sample_rate = librosa.load(file_path, sr=16000)  # Ensure sample rate matches model expectations
                mel_spec = librosa.feature.melspectrogram(y=audio_input, sr=sample_rate, n_mels=80)  # Adjusted mel bands
                
                # Debugging: print shape of mel spectrogram
                print(f"Shape of mel spectrogram for {filename}: {mel_spec.shape}")
                
                mel_spec = mel_spec.T  # Transpose mel spectrogram to match input shape
                mel_input = torch.tensor(mel_spec).unsqueeze(0).to(device)  # Add batch dimension
                
                # Debugging: print the shape after adding batch dimension
                print(f"Shape of mel spectrogram after adding batch dimension: {mel_input.shape}")
                
                # Ensure mel spectrogram is in the correct format (batch, mel_bins, time_steps)
                mel_input = mel_input.transpose(1, 2)  # Transpose to match (batch, mel_bins, time_steps)
                
                # Debugging: check the new shape
                print(f"Shape of mel spectrogram after transposing for model: {mel_input.shape}")
                
                # Calculate the length of the input
                input_length = torch.tensor([mel_input.size(2)]).to(device)  # Number of frames in the mel spectrogram
                print(f"Input length for {filename}: {input_length}")
                
            except Exception as e:
                print(f"Error preprocessing audio {filename}: {e}")
                continue
            
            # Pass the audio through the model's forward method to get the logits
            try:
                logits, _ = model.encoder(audio_signal=mel_input, length=input_length)  # Unpack the tuple returned by the model
                logits = logits.squeeze(0)  # Remove batch dimension
            except Exception as e:
                print(f"Error processing logits for {filename}: {e}")
                continue
            
            # Get word-level timestamps and confidence scores
            word_times_and_confidence = get_word_level_timestamps_and_confidence(logits)
            
            # Write the filename, transcription, and word-level timestamps with confidence to the output file
            f.write(f"{filename}:\n")
            for entry in word_times_and_confidence:
                f.write(f"Word: {entry['word']}, Start: {entry['start_time']:.2f}s, End: {entry['end_time']:.2f}s, Confidence: {entry['confidence']:.2f}\n")
            f.write("\n")
            print(f"Processed {filename}")
    
    print(f"Transcriptions with timestamps saved to {output_file}")

