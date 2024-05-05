import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clear_writting(writting):
    """
        Limpa todas as setenças inseridas.
    """

    # Tokeniza todas as frases inseridas, lematiza cada uma delas e retorna

    sentence_words = nltk.word_tokenize(writting)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(writting, words):
    """
        Pega as setenças que são limpas e cria um pacote de palavras
        que são usadas para classes de previsão que são baseadas nos resultados obtidos
    """

    sentence_words = clear_writting(writting)

    bag = [0] * len(words)
    for setence in sentence_words:
        for i, word in enumerate(words):
            if word == setence:

                # Atribui 1 no pacote de palavra se a palavra atual estiver na posição da sentença

                bag[i] = 1
    return (np.array(bag))

def class_prediction(writing, model):
    """
        Faz a previsão do pacote de palavras, usando como limite de erro 0.25
        e classificando esses resultados por força da probabilidade.
    """

    # Filtrando as previsões abaixo de um limite 0.25

    prevision = bag_of_words(writing, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction) if response > 0.25]

    '''
        Verificando nas previsões se não há 1 na lista, se não houver, envia a resposta padrão (anything_else)
        ou se não corresponde a margem de erro
    '''
    
    if not np.any(prevision == 1) or len(results) == 0:
        results = [[0, response_prediction[0]]]

    # Classifica por força de probabilidade

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents, intents_json):
    """
        Pega a lista gerada e verifica o arquivo JSON e produz a maior parte das respostas com a maior probabilidade.
    """

    tag = intents[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:

            # Caso as respostas sejam um array contendo mais de uma, usa-se a função de random para pegar uma resposta randomica da lista
            
            result = random.choice(i["responses"])
            break

    return result