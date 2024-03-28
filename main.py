from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_hastie_10_2, make_classification
import pandas as pd
import plotly.express as px

##Data Cleaning##
df = pd.read_csv('obesity.csv', encoding = "ISO-8859-1")
df.dropna()

# fumambebem = df[(df['SMOKE'] == 1) & (df['CALC'] > 0 ) & (df['family_history_with_overweight'] == 1 )]

fumambebem = df #df[(df['SMOKE'] == 1) & (df['CALC'] > 0 ) & (df['family_history_with_overweight'] == 1) & (df['FAVC'] == 1) & (df['FCVC'] == 0) & (df['NCP'] == 2) & (df['CAEC'] >= 2)]
print(df)

normaisfumambebem = df[(df['NObeyesdad'] == 0) & (df['SMOKE'] == 1) & (df['CALC'] > 0 )]

normaisfumam = df[(df['NObeyesdad'] == 0) & (df['SMOKE'] == 1)]

normais = df[df['NObeyesdad'] == 0]

obesosfumambebem = df[(df['NObeyesdad'] == 1) & (df['SMOKE'] == 1) & (df['CALC'] > 0 )]

obesosfumam = df[(df['NObeyesdad'] == 1) & (df['SMOKE'] == 1)]

obesos = df[df['NObeyesdad'] == 1]

df.head()

# print(len(obesosfumambebem))
# print(len(normaisfumambebem))


##Boxplots##
# Criar um boxplot das idades apenas para os obesos
plt.figure(figsize=(8, 6))
obesos.boxplot(column='Age')
plt.title('Boxplot de Idades para Obesos')
plt.ylabel('Idade')
plt.xlabel('Obesos')

plt.show()

# Criar um boxplot das idades apenas para os normais
plt.figure(figsize=(8, 6))
normais.boxplot(column='Age')
plt.title('Boxplot de Idades para Normais')
plt.ylabel('Idade')
plt.xlabel('Normais')

plt.show()


##Scatter Plot
obesos = df[df['NObeyesdad'] == 1]
nao_obesos = df[df['NObeyesdad'] == 0]

fig = px.scatter()
fig.add_scatter(x=[1]*len(obesos), y=obesos['Age'], mode='markers', name='Obesos')
fig.add_scatter(x=[0]*len(nao_obesos), y=nao_obesos['Age'], mode='markers', name='Não Obesos')

fig.update_layout(title='Idade dos Obesos vs Não Obesos',
                  xaxis_title='Idade',
                  yaxis_title='fumambebem (1=Obeso, 0=Não Obeso)',
                  width=800)

fig.show()


##Treinamento e Validação Classificação##
Y = (fumambebem['NObeyesdad'] == 1)  
X = fumambebem.drop('NObeyesdad', axis=1) 


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#20 47 : 0.84 | 0.42857142857142855
#10 55 : 0.8214285714285714 | 0.5
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



##Decision Tree no ScikitLearn Classificação##
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


exemplo_pessoa = pd.DataFrame({'Gender': [1], 'Age': [3], 'family_history_with_overweight': [1], 'FAVC': [1], 'FCVC': [1], 'NCP': [1], 'CAEC': [1], 'SMOKE': [0], 
                               'CH2O': [0], 'SCC': [0], 'FAF': [0], 'TUE': [0], 'CALC': [1], 'Automobile': [1], 'Bike': [1], 'Motorbike': [1], 
                               'Public_Transportation': [1], 'Walking': [1]})

obeso = model.predict(exemplo_pessoa)
print("Previsão para a pessoa:", obeso)
# salva o modelo treinado para uso posterior
dump(model, 'desafio.joblib')