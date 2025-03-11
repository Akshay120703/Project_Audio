🔹 1. Beam Size (beam_size)
Effect: Controls the number of beams (hypotheses) explored during decoding.
Default: 5
Optimal Range: 5-10
Higher Values: Improve accuracy but increase computational cost.

🔹 2. Temperature (temperature)
Effect: Adds randomness to token selection. Lower values make the model more deterministic, while higher values make it more diverse.
Default: 0.0
Optimal Range: 0.0-0.5
Lower Values (0.0-0.3): More stable, but can cause repetitive errors.
Higher Values (0.4-0.7): Increases variation in outputs, helpful for informal speech.

🔹 3. Best of (best_of)
Effect: Generates multiple hypotheses and picks the best-scoring one.
Default: 5
Optimal Range: 3-10
Higher Values: Improve accuracy at the cost of increased processing time.

🔹 4. Patience (patience)
Effect: Affects how much Whisper waits before finalizing a token.
Default: 1.0
Optimal Range: 1.0-2.0
Higher Values: Allow more refinement but increase processing time.

🔹 5. Compression Ratio Threshold (compression_ratio_threshold)
Effect: Filters out results with high compression ratios (indicating repetitive/unintelligible text).
Default: 2.4
Lower Values (1.5-2.0): Filter out more bad outputs.
Higher Values (3.0+): Allow more outputs but risk gibberish.

🔹 6. Log Probability Threshold (logprob_threshold)
Effect: Filters out words with low confidence scores.
Default: -1.0
Optimal Range: -1.0 to -0.3
Higher Values (-0.3 to 0.0): More aggressive filtering of uncertain words.

🔹 7. No Speech Probability Threshold (no_speech_threshold)
Effect: Determines when the model decides if no speech is present in audio.
Default: 0.6
Lower Values (0.2-0.5): Reduce false positives where silence is mistakenly transcribed.
Higher Values (0.7-1.0): Prevent skipping of very low-volume speech.

🔹 8. Language (language)
Effect: Forces a specific language instead of auto-detection.
Default: Auto
Setting It Manually: Avoids misclassification errors in multilingual audio.

🔹 9. Prompt (initial_prompt)
Effect: Allows conditioning on prior text for better context-aware transcription.
Usage: Provide domain-specific keywords, phrases, or context.
Example: initial_prompt="technical jargon, AI, machine learning" improves transcriptions of tech-related content.

🔹 10. Condition on Previous Text (condition_on_previous_text)
Effect: Controls whether the model considers prior context in segment generation.
Default: True
Setting to False: Reduces hallucinations but may lose context.

🔹 11. Word Timestamps (word_timestamps)
Effect: Enables word-level timestamps for precise alignment.
Default: False
Setting to True: Essential for subtitle generation and AI dubbing.

🔹 12. Task (task)
Effect: Controls whether the model transcribes or translates.
Options: "transcribe" (default) or "translate"
Translation Mode: Converts speech to English, useful for multilingual audio.

🔹 13. Suppress Tokens (suppress_tokens)
Effect: Prevents specific tokens from appearing in the output.
Usage: Prevent unwanted characters, numbers, or filler words.
