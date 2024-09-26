## Online Noise Removal Tools

https://app.audostudio.com/


## Noisy datasets

https://labs.freesound.org/datasets/

https://research.google.com/audioset/ontology/index.html




## Assembly.ai API

Code Part:- # `pip3 install assemblyai` (macOS)
f# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "450d321e71fb4436a9a15e6cede57b44"
transcriber = aai.Transcriber()

transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
transcript = transcriber.transcribe("./my-local-audio-file.wav")

print(transcript.text)



## Resources

https://research.ijcaonline.org/volume45/number11/pxc3879438.pdf

https://www.isca-archive.org/interspeech_2021/kashyap21_interspeech.pdf

https://github.com/Dhriti03/Noise-Reduction/blob/master/Blog.pdf


https://www.kaggle.com/code/faressayah/signal-processing-with-python#Denoising-Data-with-FFT

https://www.kaggle.com/code/sahandissanayaka/complete-notebook-for-audio-preprocessing
