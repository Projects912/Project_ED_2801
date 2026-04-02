Project Overview

This project is a smart AI system that detects emotions and sentiments from text, audio, and video. It not only predicts what someone is feeling but also gives visual insights into how emotions are expressed, especially through the body in videos.

--------------------------------------------------------------------------------------------------------------------------------------------------

Data Splitting

-----------------

Training: 70%

Testing: 20%

Validation: 10%


Dataset Creation/
config.py
     Contains all hyperparameters, file paths are used across the project.
dataset.py
     Data Preprocessing 


The system uses three types of inputs:

Text (Speech Transcripts) 

The text is taken from CSV files with the transcript of each dialogue.
 
Transformed into numerical embeddings and passed to the model with the help of DeBERTa tokenizer. 

Audio (Speech) 

The audio files are stored as a .wav file in the dataset folder. 

Processed using Wav2Vec2 to obtain features such as intonation, tone, and patterns of speech. 

Fixed length is used for consistency; the missing audio are set to zeros. 

Visual (Frames from Video) 

The uploaded MP4 file (your video.mp4) is converted into video frames. 

The faces are recognized and the prediction of emotion is done with the help of FER library. 

Incomplete data is filled by using zero tensors instead of missing faces or frames.
  

------------------------------------------------------------------------------------------------------------

Model/
model.py
     Model architecture,including Feature extractors,Fusion modules,Attention layers,and Multitask heads.
train.py
     Contains functions for training train_epoch and evaluate,including loss calculation and metric computation.

--------------------------------------------------------------------------------------------------------------
main.py

Loads datasets and dataloaders,Initializes the model, loss functions, and optimizer Runs the training loop with validation
Saves the best model 
Evaluates on the test set 

-----------------------------------------------------------------------------------------------------------------

torch==2.0.1

torchvision==0.15.2

transformers==4.31.0

pandas==2.0.3

numpy==1.24.4

soundfile==0.12.1

scikit-learn==1.2.2

tqdm==4.66.1

Pillow==10.0.0
