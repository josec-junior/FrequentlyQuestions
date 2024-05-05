import json
import pickle
import nltk
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# Inicializando a lista de palavras, classes, documentos e definindo quais palavras serão ignoradas

words = []
documents = []

# Fazendo a leitura do arquivo intents.json e transformando em JSON

intents = json.load(open("intents.json","r", encoding="utf8"))

# Adicionando as tags à lista de classes

classes = [i["tag"] for i in intents["intents"]]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# # Percorrendo o array de objetos

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Realizando tokenização dos patterns com ajuda do nltk e adicionando-os à lista de palavras
        
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # Adicionando aos documentos para poder indentificar a tag para a mesma

        documents.append((word, intent["tag"]))

# Lematização das palavras, ignorando as palavras da lista ignore_words

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Classificação das listas

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Deep Learning

# Salvando as palavras e classes nos arquivos pkl

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Inicializando o treinamento

training = []
output_empty = [0] * len(classes)
for document in documents:

    # Inicializando o "saco" de palavras
    bag = []

    # Listando as palavras do pattern

    pattern_words = document[0]

    # Lematizeando cada palavram, na tentativa de representar palavras relacionadas

    pattern_words = [lemmatizer.lemmatize( word.lower() ) for word in pattern_words]

    # Criando conjunto de palavras com 1, se a correspondência de palavras for encontrada no padrão atual

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output_row atuará como uma chave para a lista, onde a saida será 0 para cada tag e para a tag atual
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Embaralhando o conjunto de treinamentos e transformando em numpy array
random.shuffle(training)
training = np.array(training)

# Criando lista de treino, sendo x os patterns e y as intenções

x = list(training[:, 0])
y = list(training[:, 1])

# Criando o modelo com 3 camadas.
## Primeira camada de 128 neurônios,
## Segunda camada de 64 neurônios e terceira camada de saída
## contém número de neurônios igual ao número de intenções para prever a intenção de saída com softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation="softmax"))

# Compilação do modelo com descida de gradiente estocástica com gradiente acelerado de Nesterov.
# A ideia da otimização do Momentum de Nesterov, ou Nesterov Accelerated Gradient (NAG)
# é medir o gradiente da função de custo não na posição local,
# mas ligeiramente à frente na direção do momentum.
# A única diferença entre a otimização de Momentum é que o gradiente é medido em θ + βm em vez de em θ.

# sgd = optimizers.SGD(lr=0.001, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])


# Ajustando e salvando o modelo

m = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save("model.h5", m)

print("Fim")