import json
from tkinter import *
from extract import class_prediction, get_response
from tensorflow.keras.models import load_model

# Extraindo o modelo usando o keras

model = load_model("model.h5")

# Carregando as intenções

intents = json.load(open("intents.json","r", encoding="utf8"))


base = Tk()
base.title("FrequentlyQuestions")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

def chatbot_response(msg):
    """
        Resposta do bot
    """

    ints = class_prediction(msg, model)
    res = get_response(ints, intents)
    return res

def send():
    """
        Envia a mensagem
    """
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        # Chat.config(foreground="#000000", font="(Arial, 12)")

        response = chatbot_response(msg)
        Chat.insert(END, f"FrequentlyQuestions: {response}\n\n")

        Chat.config(state=DISABLED)
        Chat.yview(END)
    
# Criando a janela do chat

Chat = Text(base, bd=0, bg="white", height="8", width="100", font="Arial")
Chat.config(state=DISABLED)

# Vinculando a barra de rolagem à janela de bate-papo

scrollbar = Scrollbar(base, command=Chat.yview)
Chat["yscrollcommand"] = scrollbar.set

#Criando o botão de envio de mensagem, onde o comando envia para a função de send

SendButton = Button(base, font=("Verdana", 10, "bold"),
text="Enviar", width="12", height=2, bd=0, bg="#09b835",
activebackground="#333", fg="#FFFFFF", command=send)

# Criando o box de texto

EntryBox = Text(base, bd=0, bg="white", width="29", height="2", font="Arial")

# Colocando todos os componentes na tela

scrollbar.place(x=376, y=6, height=386)
Chat.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=260)
SendButton.place(x=6, y=401, height=50)

base.mainloop()