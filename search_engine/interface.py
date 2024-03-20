import streamlit as st
import pandas as pd
from test_search_esp import input
from test_search_en import input_en

df = pd.read_csv("articles_paragraphs_sin_referencias.csv")

# Links list
titulos = df['article_name'].unique()
titles_list = titulos.tolist()
titles_list.sort()
df_titles = pd.DataFrame(titles_list)

# Links to add to the dataframe
links = ['https://equinoxailab.ai/ai-for-more-competitive-retailers/', 'https://equinoxailab.ai/cuando-el-deep-learning-es-la-mejor-opcion/', 'https://equinoxailab.ai/neural-networks-vs-human-brain/', 'https://equinoxailab.ai/bi-for-making-decisions/', 'https://equinoxailab.ai/people-and-object-detection-with-jetson-and-python/', 'https://equinoxailab.ai/despliegue-y-arquitecturas-de-solucion-en-ai/', 'https://equinoxailab.ai/diseno-ux-y-ai-confianza-y-reciprocidad/', 'https://equinoxailab.ai/ai-helps-weather-forecast/', 'https://equinoxailab.ai/en/enhance-medicine-using-ai/', 'https://equinoxailab.ai/graphics-processing-units-gpu-vs-fpga/', 'https://equinoxailab.ai/la-inteligencia-artificial-permite-ser-mas-humanos/',
         'https://equinoxailab.ai/interfaces-invisibles/', 'https://equinoxailab.ai/its-all-about-data/', 'https://equinoxailab.ai/inteligencia-artificial-en-beneficio-a-la-humanidad/', 'https://equinoxailab.ai/es/modelos-matematicos-detras-de-machine-learning/', 'https://medium.com/@equinoxailab/medical-utopias-between-design-and-ai-b5409ba06535', 'https://equinoxailab.ai/nlp-understanding-machines/', 'https://equinoxailab.ai/quantum-computing-vs-old-computer-problems/', 'https://equinoxailab.ai/rpa-invest-time-valuable-tasks/', 'https://equinoxailab.ai/random-number-generators/', 'https://equinoxailab.ai/robotics-driven-by-artificial-intelligence/', 'https://equinoxailab.ai/shaping-the-metaverse/', 'https://equinoxailab.ai/simulations-with-quantum-chemistry/']

df_titles['link'] = links
df_titles.columns = ['article', 'link']


# Streamlit

tab1, tab2 = st.tabs(["English", "Español"])

with tab1:
    st.title('Paragraph Searcher')
    search_input = st.text_input(
        'Introduce your input for the search', placeholder='Search input')
    t_en, p_en, p_no_en, t_en2, p_en2, p_no_en2, t_en3, p_en3, p_no_en3 = input_en(
        search_input)
    if (search_input != ''):
        df_titles1 = df_titles.loc[df_titles['article'] == t_en]
        link = df_titles1.loc[df_titles1['article'] == t_en, 'link'].iloc[0]
        st.markdown('Here you have the top 3 coincidences for your input:')
        st.markdown('---')
        st.markdown(f'Paragraph {p_no_en} from the article:')
        st.markdown(f'> *"{t_en}"*')
        st.markdown('Here you have it:')
        st.markdown(f'>*{p_en}*')
        st.markdown(
            f'**You can find the full article in this link :** *{link}*')
        st.markdown('---')
        df_titles2 = df_titles.loc[df_titles['article'] == t_en2]
        link = df_titles2.loc[df_titles2['article'] == t_en2, 'link'].iloc[0]
        st.markdown(f'Paragraph {p_no_en2} from the article:')
        st.markdown(f'> *"{t_en2}"*')
        st.markdown('Here you have it:')
        st.markdown(f'>*{p_en2}*')
        st.markdown(
            f'**You can find the full article in this link :** *{link}*')
        st.markdown('---')
        df_titles3 = df_titles.loc[df_titles['article'] == t_en3]
        link = df_titles3.loc[df_titles3['article'] == t_en3, 'link'].iloc[0]
        st.markdown(f'Paragraph {p_no_en3} from the article:')
        st.markdown(f'> *"{t_en3}"*')
        st.markdown('Here you have it:')
        st.markdown(f'>*{p_en3}*')
        st.markdown(
            f'**You can find the full article in this link :** *{link}*')

with tab2:
    st.title('Buscador de párrafos')
    search_input = st.text_input(
        'Introduce el texto a buscar', placeholder='Texto a buscar')
    t_es, p_es, p_no_es, t_es2, p_es2, p_no_es2, t_es3, p_es3, p_no_es3 = input(
        search_input)
    if (search_input != ''):
        df_es1 = df_titles.loc[df_titles['article'] == t_es]
        link = df_es1.loc[df_es1['article'] == t_es, 'link'].iloc[0]
        st.markdown('Aquí tienes las 3 mejores coincidencias con tu input')
        st.markdown('---')
        st.markdown(f'Párrafo {p_no_es} del artículo:')
        st.markdown(f'> *"{t_es}"*')
        st.markdown(f'>*{p_es}*')
        st.markdown(
            f'**Puedes encontrar el artículo completo en el siguiente enlace :** *{link}*')
        st.markdown('---')
        df_es2 = df_titles.loc[df_titles['article'] == t_es2]
        link = df_es2.loc[df_es2['article'] == t_es2, 'link'].iloc[0]
        st.markdown(f'Párrafo {p_no_es2} del artículo:')
        st.markdown(f'> *"{t_es2}"*')
        st.markdown(f'>*{p_es2}*')
        st.markdown(
            f'**Puedes encontrar el artículo completo en el siguiente enlace :** *{link}*')
        st.markdown('---')
        df_es3 = df_titles.loc[df_titles['article'] == t_es3]
        link = df_es3.loc[df_es3['article'] == t_es3, 'link'].iloc[0]
        st.markdown(f'Párrafo {p_no_es3} del artículo:')
        st.markdown(f'> *"{t_es3}"*')
        st.markdown(f'>*{p_es3}*')
        st.markdown(
            f'**Puedes encontrar el artículo completo en el siguiente enlace :** *{link}*')
        st.markdown('---')
