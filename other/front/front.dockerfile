# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le code source de l'application dans le répertoire de travail
COPY . .


# Installer les dépendances à partir du fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "frontend_for_render.py"]
