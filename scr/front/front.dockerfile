# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le code source de l'application dans le répertoire de travail
COPY . .


# Installer les dépendances à partir du fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install spacy

# Installer les modèles spaCy pour les langues anglaise et française
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download xx_sent_ud_sm



# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "frontend.py"]