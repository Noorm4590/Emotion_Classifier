import os
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
import torch.nn.functional as F
import streamlit as st

# Define the list of emotion attributes
attributes = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

# Define the Emotion Sentence Classifier model
class Emotion_Sentence_Classifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, 64)
        self.classifier = nn.Linear(64, self.config['n_labels'])
        nn.init.xavier_uniform_(self.classifier.weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
        self.dropout = nn.Dropout()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        logits = self.classifier(pooled_output)
        loss = 0
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
        return loss, logits

# Load model configuration
config = {
    'model_name': 'distilroberta-base',
    'n_labels': len(attributes),
    'batch_size': 128,
    'lr': 1.5e-6,
    'n_epochs': 10
}

# Initialize the model
model = Emotion_Sentence_Classifier(config)

# Load the model state dictionary
model_state_dict = torch.load('./Em_model_22_1.5e-06lr.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

# Define the inference function
def inference(sentence):
  x = tokenizer(sentence)
  with torch.no_grad():
    _ , output = model(torch.tensor([x['input_ids']]),torch.tensor([x['attention_mask']]),None)
    prediction = torch.where(F.sigmoid(output) > 0.5,torch.tensor(1),torch.tensor(0))
    em = []
    for index , el in enumerate(prediction[0]):
      if el == 1:
        em.append(attributes[index])
    return em

# Streamlit UI
st.title("Emotion Classifier")
st.write("Enter a sentence to predict the associated emotions.")

# Input text box
user_input = st.text_input("Input Sentence:", "I don't want to talk about it!")

# Predict button
if st.button("Analyze"):
    emotions = inference(user_input)
    if emotions:
        st.write("### Detected Emotions:")
        for emotion in emotions:
           st.write(f"- {emotion}")
        # st.write(", ".join(emotions))
    else:
        st.write("No emotions detected.")


# Add contact information to the sidebar
st.sidebar.markdown("""
## Contact Information

**Developed by Muhammad Mubashir**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/muhammad-mubashir-4441072ba/)

[![Email](https://img.shields.io/badge/Email-Send%20an%20Email-red?style=for-the-badge&logo=gmail)](mailto:mubashir.dev.02@gmail.com)
""")

# Add some styling to the sidebar
st.sidebar.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 10px;
}
.sidebar .sidebar-content a {
    text-decoration: none;
}
.sidebar .sidebar-content h2 {
    font-family: 'Poppins', sans-serif;
    color: #2b2b2b;
}
.sidebar .sidebar-content p {
    font-family: 'Poppins', sans-serif;
    color: #2b2b2b;
}
</style>
""", unsafe_allow_html=True)