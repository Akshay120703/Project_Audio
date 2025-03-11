Prompt:
You are given transcriptions from two different ASR models, Whisper-hindi-large-v2 and IndicConformer. Your task is to ensemble these outputs into a single, high-quality transcript while maintaining proper word order and coherence.

Instructions:
Preserve All Words: Ensure that no word is omitted. The final transcript must contain all words from both models.

Use Word Timestamps: The Whisper model provides timestamps for each word. Use these timestamps to determine the correct sequence of words.

Maintain Proper Order: The final transcript should have a natural flow, ensuring that words appear in the correct grammatical and contextual sequence.

Select the Best Word: If both models provide a word for the same timestamp range, select the most accurate word based on context and meaning. Use the confidence score from the Whisper model to help decide if needed.

Avoid Repetitions: Do not repeat words unnecessarily. If a word is present in both transcripts in the same position, choose the best one rather than duplicating it.

Handle Unaligned Words: If a word appears in only one model's output, place it at the most appropriate position in the sentence while maintaining meaning.

Ensure Fluency & Readability: The final transcript should be well-structured, grammatically correct, and easily readable.

Input Format:
Whisper-hindi-large-v2 Output: A list of words with their corresponding start time, end time, and confidence score.
IndicConformer Output: A continuous transcript without timestamps.

Example Input:
Whisper Output (Word-Level Timestamps):

Word,Start Time,End Time,Confidence
उसे,0.0,1.22,0.504
काम,1.22,1.5,0.825
करवाती,1.5,1.92,0.786
है,1.92,2.1,0.882
...
IndicConformer Output:

बीती है उससे काम करवाती है इससे उसका शारीरिक विकास नहीं हो पता है...
Expected Output:
A combined transcript that merges the words correctly, ensuring all words are retained while maintaining the logical structure of the sentence. The final output should be a single, properly structured transcript that reads naturally and fluently.

