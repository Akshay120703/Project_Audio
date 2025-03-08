import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel

# ========== Load IndicConformer Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conformer_model = EncDecCTCModel.restore_from(restore_path="/home/pc/Desktop/IndicConformer/ai4b_indicConformer_hi.nemo")
conformer_model.freeze()
conformer_model = conformer_model.to(device)
conformer_model.cur_decoder = "ctc"  # Set to CTC decoding

# ========== Load Wav2Vec2 Model for Alignment ==========
processor = Wav2Vec2Processor.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("/home/pc/Desktop/IndicConformer/Wav2Vec").to("cpu")

# ========== Function to Load Audio ==========
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    print(f"Audio loaded. Length: {len(waveform)} samples, Sample rate: {sr} Hz.")
    return waveform

# ========== Function to Extract Audio Between Frames ==========
def extract_audio_between_frames(audio, frame_start, frame_end, frame_duration=0.02, target_sr=16000):
    # Convert frame number to sample index
    frame_start_sample = int(frame_start * frame_duration * target_sr)
    frame_end_sample = int(frame_end * frame_duration * target_sr)
    
    # Slice the audio to get the segment between the frames
    audio_segment = audio[frame_start_sample:frame_end_sample]
    print(f"Extracted audio from {frame_start} to {frame_end} frames.")
    
    return audio_segment

# ========== Function to Get Transcript for Audio Segment ==========
def get_transcript_for_audio_segment(audio_segment):
    # Pass the audio segment through the IndicConformer model for transcription
    transcript = conformer_model.transcribe([audio_segment], batch_size=1)[0]
    return transcript

# ========== Main Processing ==========
audio_file = "/home/pc/Desktop/audio/dir/01-00008-03.mp3"  # Change to your audio file path
frame_start = 0  # Example: Start frame
frame_end = 100    # Example: End frame

# Load the audio
audio = load_audio(audio_file)

# Extract the audio segment between the specified frames
audio_segment = extract_audio_between_frames(audio, frame_start, frame_end)

# Get the transcript for the extracted audio
transcript = get_transcript_for_audio_segment(audio_segment)

print(f"Transcript between frames {frame_start} and {frame_end}: {transcript}")

