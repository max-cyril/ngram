import streamlit as st
import requests
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from langdetect import detect,detect_langs
import spacy


nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
nlp_other= spacy.load('xx_sent_ud_sm')

#background
sdb_bg_img = """
    <style>
    [data-testid="stSidebarContent"]
    {
    background-color: #fcfafc;
    opacity: 1;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #fcfafc 4px ), repeating-linear-gradient( #dad1c955, #dad1c9 );</style>
    """
st.markdown(sdb_bg_img, unsafe_allow_html=True)

# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"]
# {
# background-color: #fcfafc;
# opacity: 1;
# background-image:  radial-gradient(#227b75 0.4px, transparent 0.4px), radial-gradient(#227b75 0.4px, #fcfafc 0.4px);
# background-size: 16px 16px;
# background-position: 0 0,8px 8px;
# </style>
# """
# st.markdown(page_bg_img, unsafe_allow_html=True)





# Interface utilisateur Streamlit
st.title("Next Word Prediction")


    
# Entrée du corpus pour l'entraînement
corpus_input = st.text_area("Enter corpus for training (each sentence on a new line):",help="About 10 minutes for 500 words to train")

if corpus_input:
    text_count = ' '.join(corpus_input.split())
    st.write("you enter :",len(text_count.split()),"\n", "words")
    st.write("N-GRAM  Neural Network model training will  last About" , round(len(text_count.split())*(1/50)), " ", "minutes")
    
    # Détection de la langue du corpus
    language_deducted = detect_langs(text_count)
    language = detect(text_count)
    lang_data = {"Language": [str(lang.lang) for lang in language_deducted], "Probability": [lang.prob for lang in language_deducted]}
    #st.write("Detected Languages:")
    #st.dataframe(lang_data)
    
    
    # Reconnaissance des entités nommées
    if language == 'en':
        doc = nlp_en(text_count)
    elif language == 'fr':
        doc = nlp_fr(text_count)
    else:
        doc = nlp_other(text_count)
        #st.write("NER model for detected language is not available.")
    
    if doc:
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        df_entities = {"Entity": [ent[0] for ent in entities], "Tag": [ent[1] for ent in entities]}
        #st.write("Named Entities:")
        #st.dataframe(df_entities)
        
    

    # Affichage des tableaux côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.write("Detected Languages:")
        st.dataframe(lang_data)
    with col2:
        if doc:
            st.write("Named Entities:")
            st.dataframe(df_entities, height=150)


        
# Bouton pour démarrer l'entraînement
if st.button("Train Model"):
    corpus = corpus_input.split('\n')
    data = {"corpus": corpus}
    
    with st.spinner('Training in progress...'):
        response = requests.post("http://ngram_api:8000/train/", json=data)
        result = response.json()
        st.write(result["message"])
        
        progress_bar = st.progress(0)
        for i, loss in enumerate(result["progress"]):
            progress_bar.progress((i + 1) / len(result["progress"]))
            if i == 999:
                st.write(f"Epoch {i + 1}/{len(result['progress'])}: Loss = {loss}")
                time.sleep(0.1)  # Just to simulate the delay




with st.sidebar:
    if corpus_input:
        # Génération du WordCloud
        text = ' '.join(corpus_input.split('\n'))
        wordcloud = WordCloud(width=300, height=600, background_color='white').generate(text)
        
        # Affichage du WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.sidebar.write("WorldCloud Corpus")
        st.sidebar.pyplot(plt)

# Entrée du contexte pour la prédiction
context = st.text_input("Enter a context (at least 2 words):", "")

# Obtenir les prédictions en utilisant uniquement les deux derniers mots du contexte
if context:
    # Séparer les mots du contexte
    context_words = context.split()
    # Utiliser les deux derniers mots pour la prédiction
    if len(context_words) >= 2:
        context_to_predict = ' '.join(context_words[-2:])
        # Appeler l'API pour obtenir les prédictions
        response = requests.post("http://ngram_api:8000/predict/", json={"context": context_to_predict})
        predictions = response.json()["predictions"]
        st.write("Predictions:", predictions)
    else:
        st.write("Enter at least 2 words to predict the next")
