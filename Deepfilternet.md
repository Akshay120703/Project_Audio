I have found a new Noise Suppression Deep Learning Model named "DeepFilterNet".

I have tested the results on the small version and it gave much better results than the noisereduce library.

There are many advantages of DeepFilternet over Noisereduce:-

1) Better clarity and enhancement of speaker voice.
2) Used deep layered advanced filters for selective segregation and separation of speaker voice and background noise.(It properly identifies speaker voice and includes only that portion for audio enhancement pipeline and rest of the portion is denoised)
3) Better identification on all types of noises. Provides special feature of adding more noisy/clean datasets for further improvement in noise reduction categories.
4) Very less portion is cut-out or removed in comparison to noisereduce library.
5) Better adaptability to dynamic/variable noise
6) Better control over amplitude variation
7) Higher processing rate with low sample size reduces loss of information.

I have simply tested the small model on Google Colab with T4-GPU and the results were quite better than the noise reduction libraries.

The Colab Notebook with a small testing code for small model of DeepFilterNet only for single file processing is provided below: https://colab.research.google.com/drive/1VF4yPWvk2FLl8IygDIR88HGQpUKiu0Eq?usp=sharing

The Model suits best for batch file processing.

The comparison between DeepFilterNet(small model) and Noisereduce can be clearly seen in the Noise Profile Screenshot from Audacity
Image_Link:- https://drive.google.com/file/d/1TxZVl-OsOTDnUeQcPnx5jZPvRxwoY_NJ/view?usp=sharing

Audio Labels in Audacity

File1:- Original Audio(Link:- https://drive.google.com/file/d/1CBvh3fT7UmLq3UDq6C13r8GvXDdbCDja/view?usp=sharing)

File2:- DeepFilterNet Enhanced Audio(Link:- https://drive.google.com/file/d/1yDIGCsn-zsRSus4CP6-d9VDTNIjQEK4Q/view?usp=sharing)

File3:- NoiseReduce Enhanced Audio(Link:- https://drive.google.com/file/d/1yWeTsZLceF4mTEMSof0FhBBvVqz8tgN-/view?usp=sharing)
