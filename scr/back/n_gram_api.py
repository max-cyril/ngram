"""Main module."""
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import uvicorn
from typing import List

# Définition du modèle NGram
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Déclaration de l'application FastAPI
app = FastAPI()

# Modèles de données
class TrainingData(BaseModel):
    corpus: List[str]

class PredictionData(BaseModel):
    context: str

# Variables globales pour le modèle, le vocabulaire et les indices
model = None
word_to_ix = {}
ix_to_word = []

@app.post("/train/")
async def train_ngram_model(data: TrainingData):
    global model, word_to_ix, ix_to_word
    
    # Concaténation des phrases du corpus
    text = ' '.join(data.corpus)
    
    # Tokenisation du texte
    words = text.split()
    vocab = set(words)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    
    # Création des trigrammes
    trigrams = [((words[i], words[i + 1]), words[i + 2]) for i in range(len(words) - 2)]
    
    # Paramètres du modèle
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Entraînement du modèle
    num_epochs = 1000
    progress = []

    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in tqdm(trigrams, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        progress.append(total_loss)

    torch.save(model.state_dict(), 'ngram_model.pth')
    return {"message": "Model trained successfully.", "progress": progress}




@app.post("/predict/")
async def predict_next_word(data: PredictionData):
    global model, word_to_ix, ix_to_word
    
    if model is None or not word_to_ix or not ix_to_word:
        raise HTTPException(status_code=400, detail="Model not trained yet")

    # Extraction des mots du contexte
    context_words = data.context.split()
    
    # Utilisation des deux derniers mots pour la prédiction
    if len(context_words) < 2:
        raise HTTPException(status_code=400, detail="Context too short")

    context_to_predict = context_words[-2:]
    context_idxs = torch.tensor([word_to_ix.get(w, 0) for w in context_to_predict], dtype=torch.long)
    
    with torch.no_grad():
        log_probs = model(context_idxs)
    
    top_indices = torch.argsort(log_probs, descending=True).squeeze()[:2]
    predictions = [ix_to_word[i.item()] for i in top_indices]
    
    if not predictions:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {"predictions": predictions}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)