import pandas as pd
import string
import spacy
import gensim
import numpy as np
from gensim.models import KeyedVectors


# SI TIENEN PROBLEMAS CON SPACY INSTALEN !python -m spacy download es_core_news_sm

df = pd.read_csv("articles_paragraphs_sin_referencias.csv")

df_esp = df[df['language_code'] == 'es'].reset_index()


"""
PROCESS TEXT LEMMA
"""

# load spacy nlp model
nlp = spacy.load('es_core_news_sm')

# define function for pre-processing and tokenization


def preprocess_text_lemma(text):
    """
    Preprocesses the input text by performing the following steps:
    1. Converts all characters to lowercase.
    2. Removes all punctuation.
    3. Lemmatizes the text using the spacy nlp object.
    4. Removes all stop words and short words (less than 3 characters).
    Args:
        text (str): The input text to preprocess.
    Returns:
        list: A list of preprocessed tokens.
    """
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # lemmatize
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc]
    # remove stopwords and short words
    stopwords = spacy.lang.es.stop_words.STOP_WORDS
    tokens = [
        token for token in lemmatized_text if token not in stopwords and len(token) > 2]
    return tokens


# apply pre-processing and tokenization to the 'content' column of each row
tokenized_paragraphs_lemma = []
for paragraph in df_esp['content']:
    tokens = preprocess_text_lemma(paragraph)
    tokenized_paragraphs_lemma.append(tokens)


"""
MODELO WORD 2 VEC
"""

# Train Word2Vec model
lemmaModel = gensim.models.Word2Vec(
    tokenized_paragraphs_lemma, vector_size=250, window=10, min_count=2)

# Calculate the meaning vector per paragraph
paragraph_vectors_lemma = []
for paragraph_tokens in tokenized_paragraphs_lemma:
    vectors = []
    for token in paragraph_tokens:
        if token in lemmaModel.wv.key_to_index:
            vectors.append(lemmaModel.wv[token])
    if len(vectors) > 0:
        paragraph_vectors_lemma.append(np.mean(vectors, axis=0))
    else:
        paragraph_vectors_lemma.append(np.zeros(lemmaModel.vector_size))


"""
AGREGAR LA VARIABLE DE VECTORES
"""

df_esp['vector'] = paragraph_vectors_lemma


"""
FUNCION DE COMPARACION DE COSENO ENTRE LISTA Y VECTOR
"""


def cosine_similarity_list(vectors_list, query_vector):
    """
        Computes the cosine similarity between the vector representation of the input and the vector
        representations of each sentence in the text. Returns the top-N most similar sentences, where
        N is specified by the 'n' argument.
        Args:
            vectors_list (list): A list of vector representations of the sentences in the text.
            query_vector (np.array): The vector representation of the input sentence to compare to
                the other sentences.
        Returns:
            list: A list of the top-20 most similar sentences, represented as [vector, index] pairs.
        """
    # Compute the cosine similarity between the vector representation of the input and the vector representations of each sentence in the text
    similarity_scores = []
    for vector in vectors_list:
        score = query_vector.dot(
            vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        similarity_scores.append(score)

    # Sort the sentences in descending order of their cosine similarity to the input and return the top-N most similar sentences
    n = 100
    most_similar_sentences = [[vectors_list[idx], idx] for idx in np.argsort(
        similarity_scores)[::-1][:n] if np.sum(vectors_list[idx]) != 0]

    return most_similar_sentences[:20]


"""
INPUT DEL MOTOR DE BUSQUEDA
"""


def input(userPrompt):
    """
        Given a user prompt, returns the top-3 most similar articles and paragraphs from a dataframe
        of preprocessed Spanish articles.
        Args:
            userPrompt (str): A string containing the user's input prompt.
        Returns:
            tuple: A tuple containing the title, paragraph content, and paragraph number for the
            top-3 most similar articles and paragraphs. The order of the elements in the tuple is as
            follows: (title1, paragraph1, paragraph_number1, title2, paragraph2, paragraph_number2,
            title3, paragraph3, paragraph_number3).
        """
    tokenized_prompt = preprocess_text_lemma(userPrompt)

    promptVector_lemma = np.zeros((lemmaModel.vector_size,))
    word_count = 0

    for token in tokenized_prompt:
        if token in lemmaModel.wv.key_to_index:
            promptVector_lemma += lemmaModel.wv[token]
            word_count += 1

    if word_count > 0:
        promptVector_lemma /= word_count

    var = cosine_similarity_list(df_esp['vector'], promptVector_lemma)
    titulo = df_esp["article_name"][var[0][1]]
    paragraph = df_esp["content"][var[0][1]]
    paragraph_number = df_esp["enumeration_in_article"][var[0][1]]

    titulo2 = df_esp["article_name"][var[1][1]]
    paragraph2 = df_esp["content"][var[1][1]]
    paragraph_number2 = df_esp["enumeration_in_article"][var[1][1]]

    titulo3 = df_esp["article_name"][var[2][1]]
    paragraph3 = df_esp["content"][var[2][1]]
    paragraph_number3 = df_esp["enumeration_in_article"][var[2][1]]

    return titulo, paragraph, paragraph_number, titulo2, paragraph2, paragraph_number2, titulo3, paragraph3, paragraph_number3


"""
COMPARACION DE LOS VECTORES CON EL PROMPT
"""
# RETRONA UNA LISTA DE ORDEN DE BUSQUEDA CON var[0][1] SIENDO
# PRIMERA POCISION DE LA BUSQUEDA, Y RETORNANDO SU INDEX EN EL DF DE DF_ENG


"""
FIN
"""
