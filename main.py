import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

##Data Cleaning##
df = pd.read_csv('obesity.csv', encoding = "ISO-8859-1")
df.dropna()

fumambebem = df 
normais = df[df['NObeyesdad'] == 0]
obesos = df[df['NObeyesdad'] == 1]
idadesobesos = df[df['NObeyesdad'] == 1]['Age']
idadesnaoobesos = df[df['NObeyesdad'] == 0]['Age']

# # # # # # # # # # # # # # # graficos # # # # # # # # # # # # # # #
plt.figure(figsize=(10, 6))
plt.hist([obesos['Age'], normais['Age']], bins=10, alpha=0.7, label=['Obesos', 'Não Obesos'])
plt.title('Histograma de Idades para Obesos e Não Obesos')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot([idadesobesos, idadesnaoobesos], labels=['Obesos', 'Não Obesos'])
plt.title('Boxplot de Idades para Obesos e Não Obesos')
plt.ylabel('Idade')
plt.xlabel('Categoria')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribuição de Idades')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 8))
sizes = df['NObeyesdad'].value_counts()
labels = ['Não Obeso', 'Obeso']
colors = ['lightcoral', 'lightskyblue']
explode = (0, 0.1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribuição de Obesidade')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['family_history_with_overweight'], bins=3, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Distribuição de Histórico Familiar de Sobrepeso')
plt.xlabel('Histórico Familiar de Sobrepeso (1 = Sim, 0 = Não)')
plt.ylabel('Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# # Treinamento e Validação Classificação # #
Y = (fumambebem['NObeyesdad'] == 1)  
X = fumambebem.drop('NObeyesdad', axis=1) 


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(max_depth=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_pred_train)

Y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred_test)
print()
print('##Treinamento e Validação Classificação##')
print("Precisão no treinamento:", train_accuracy)
print("Precisão no teste:", test_accuracy)
print()


# # Decision Tree no ScikitLearn Classificação # #
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(max_depth=35, random_state=12) 
model.fit(X_train, Y_train)

Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)
train_accuracy = accuracy_score(Y_train, Y_pred_train)
test_accuracy = accuracy_score(Y_test, Y_pred_test)

print()
print('##Decision Tree no ScikitLearn Classificação##')
print("Acurácia no treinamento:", train_accuracy)
print("Acurácia no teste:", test_accuracy)
print()

exemplo_pessoa = pd.DataFrame({'Gender': [1], 
                               'Age': [3], 
                               'family_history_with_overweight': [1], 
                               'FAVC': [1], 
                               'FCVC': [1], 
                               'NCP': [1], 
                               'CAEC': [1], 
                               'SMOKE': [0], 
                               'CH2O': [0], 
                               'SCC': [0], 
                               'FAF': [0], 
                               'TUE': [0], 
                               'CALC': [1], 
                               'Automobile': [1], 
                               'Bike': [1], 
                               'Motorbike': [1], 
                               'Public_Transportation': [1], 
                               'Walking': [1]
                               })

obeso = model.predict(exemplo_pessoa)

print()
print("Previsão para a pessoa: (True: OBESO | False: NÃO OBESO)", obeso)
print()

dump(model, 'desafio.joblib')