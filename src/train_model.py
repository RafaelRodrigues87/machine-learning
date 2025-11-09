import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("emails.csv") 

# Mapeamento do rótulo



#  x_treino agora recebe as mensagens, y_treino recebe os rótulos.
X_treino_msg, X_teste_msg, y_treino, y_teste = train_test_split(
    df['Mensagem'], df['Category'], test_size=0.3, random_state=43)

# 3. Transformando Texto em Números (Vectorização)
vectorizer = CountVectorizer()

# Fit e Transformar NO TREINO
X_treino_count = vectorizer.fit_transform(X_treino_msg)

# Apenas Transformar NO TESTE
X_teste_count = vectorizer.transform(X_teste_msg)


model = LogisticRegression(solver='liblinear') # Adicionado solver para evitar warnings


model.fit(X_treino_count, y_treino)


modelo_predito = model.predict(X_teste_count)


acuracia = accuracy_score(y_teste, modelo_predito)

print(f"Acurácia do Modelo: {acuracia:.4f}")


def verificar_mensagem(msg):
    msg_counts = vectorizer.transform([msg])
    previsao = model.predict(msg_counts)[0]
    return "SPAM" if previsao == 1 else "HAM"

mensagem  = input("digite sua mensagem:")

print(verificar_mensagem(mensagem))