import streamlit as st
import requests
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

sdb_bg_img = """
    <style>
    [data-testid="stSidebarContent"]
    {
    background-color: #fcfafc;
    opacity: 1;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #fcfafc 4px ), repeating-linear-gradient( #dad1c955, #dad1c9 );</style>
    """
st.markdown(sdb_bg_img, unsafe_allow_html=True)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]
{
background-color: #fcfafc;
opacity: 1;
background-image:  radial-gradient(#227b75 0.4px, transparent 0.4px), radial-gradient(#227b75 0.4px, #fcfafc 0.4px);
background-size: 16px 16px;
background-position: 0 0,8px 8px;
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Interface utilisateur Streamlit
st.title("Next Word Prediction")


# Entrée du corpus pour l'entraînement
corpus_input = st.text_area(
    "Enter corpus for training (each sentence on a new line):",
    help="About 10 minutes for 500 words to train",
)

if corpus_input:
    text_count = " ".join(corpus_input.split())
    st.write("you enter :", len(text_count.split()), "\n", "words")
    st.write(
        "N-GRAM  Neural Network model training will  last About",
        round(len(text_count.split()) * (1 / 50)),
        " ",
        "minutes",
    )

# Bouton pour démarrer l'entraînement
if st.button("Train Model"):
    corpus = corpus_input.split("\n")
    data = {"corpus": corpus}

    with st.spinner("Training in progress..."):
        response = requests.post("https://cyril-ngram.onrender.com/train/", json=data)  
        result = response.json()
        st.write(result["message"])

        progress_bar = st.progress(0)
        for i, loss in enumerate(result["progress"]):
            progress_bar.progress((i + 1) / len(result["progress"]))
            if i == 999:
                st.write(
                    f"Epoch {i + 1}/{len(result['progress'])}: Loss = {loss}"
                )
                time.sleep(0.1)  # Just to simulate the delay


with st.sidebar:
    if corpus_input:
        # Génération du WordCloud
        text = " ".join(corpus_input.split("\n"))
        wordcloud = WordCloud(
            width=300, height=600, background_color="white"
        ).generate(text)

        # Affichage du WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.sidebar.write("WorldCloud Corpus")
        st.sidebar.pyplot(plt)

# Entrée du contexte pour la prédiction
context = st.text_input("Enter a context (at least 2 words):", "")

# Obtenir les prédictions en utilisant uniquement
# les deux derniers mots du contexte
if context:
    # Séparer les mots du contexte
    context_words = context.split()
    # Utiliser les deux derniers mots pour la prédiction
    if len(context_words) >= 2:
        context_to_predict = " ".join(context_words[-2:])
        # Appeler l'API pour obtenir les prédictions
        response = requests.post(
            "https://cyril-ngram.onrender.com/predict/",
            json={"context": context_to_predict},
        )
        predictions = response.json()["predictions"]
        st.write("Predictions:", predictions)
    else:
        st.write("Enter at least 2 words to predict the next")
