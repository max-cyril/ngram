FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le contenu actuel du répertoire local dans le répertoire de travail du conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt



# Exposer le port 8000 utilisé par FastAPI
EXPOSE 8000

# Commande pour démarrer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "n_gram_api:app", "--host", "0.0.0.0", "--port", "8000"]