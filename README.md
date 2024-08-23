# Emotion Classifier - RoBERTa-Based Sentiment Analysis
## Overview
This project leverages the power of a fine-tuned RoBERTa model to classify emotions in movie lines. The model is designed to detect multiple emotions such as anger, joy, fear, and more from textual data. It’s particularly useful for applications ranging from sentiment analysis in movie reviews to enhancing personalized content recommendation systems.

## Features
Multi-label Emotion Classification: Detects multiple emotions from a single sentence.
Pre-trained RoBERTa Model: Fine-tuned on a custom movie line dataset for enhanced accuracy.
Interactive Web Interface: Built with Streamlit, allowing users to easily input sentences and receive real-time emotion analysis.
Model Performance: Achieves state-of-the-art performance in identifying emotions, demonstrating robustness and versatility.

## Usage
Enter any sentence into the input box.
Click on the "Analyze" button.
The model will predict and display the detected emotions associated with the input sentence.

## Model Architecture
The model is based on the distilroberta-base variant of RoBERTa, with a custom classification head that outputs emotion predictions. It includes the following key components:

Pre-trained RoBERTa Model: Extracts contextual embeddings from the input text.
Linear Layers: Further process the embeddings and map them to emotion classes.
Dropout and ReLU Activation: Enhance the model’s generalization ability and non-linearity.

## Example Sentences
Here are some example sentences to test the model:

"I'm thrilled about the new movie release!"
"The ending of the film left me feeling sad and empty."
"The acting was so bad, it made me angry."
