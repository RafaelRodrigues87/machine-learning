import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("emails_dataset_5k.csv")


#separa features e rotulo.
x = df[[
    "email_length", "local_length", "has_number", "num_dots",
    "num_unders", "num_hyph", "unique_ratio", "alpha_ratio",
    "digit_ratio", "domain_popular", "domain_suspicious"
]]

y = df["veracidade"]

#dividir em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

#treinando modelos
model_rf = RandomForestClassifier(n_estimators = 100, random_state= 42)
model_lr = LogisticRegression(max_iter=1000, random_state=42)

model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

#avaliando
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)


# Precisão (precision) — quantos classificados como verdadeiros realmente são.
# Revocação (recall) — quantos dos verdadeiros o modelo conseguiu achar.
# F1-score — média ponderada entre precisão e revoca
print("===== Random Forest =====")
print(classification_report(y_test, y_pred_rf))
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print("\n===== Logistic Regression =====")

# são o resumo final do classification_report — 
# ele mostra a média das métricas de desempenho 
# (precisão, recall e F1-score) de todas as classes do seu modelo.
print(classification_report(y_test, y_pred_lr))
print("Acurácia:", accuracy_score(y_test, y_pred_lr))

# Exemplo de uso: prever um novo e-mail

exemplo = {
    "email_length": 20,
    "local_length": 10,
    "has_number": 1,
    "num_dots": 1,
    "num_unders": 0,
    "num_hyph": 0,
    "unique_ratio": 0.8,
    "alpha_ratio": 0.7,
    "digit_ratio": 0.2,
    "domain_popular": 1,
    "domain_suspicious": 0
}


exemplo_df = pd.DataFrame([exemplo])
resultado = model_rf.predict(exemplo_df)[0]
print("\nPrevisao para exemplos:", "verdadeiro" if resultado == 1 else "falso")


