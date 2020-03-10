#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 21:23:04 2020

@author: xavi


Data Dictionary


Variable	Definicion	        Key
Aurvived	Supervicencia	    0 = No, 1 = Si
pclass	    Tipo de Billete	    1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex	
Age	        Edad en años	
sibsp	    Número de hermanas / cónyuges a bordo del Titanic	
#parch	    Número de padres / hijos a bordo del Titanic
#ticket	    Número de Ticket 	
#fare		Tarifa de pasajero
cabin       Número de cabina	
#embarked	Port de embarque	C = Cherbourg, Q = Queenstown, S = Southampton



Para ver todos los datos sin Truncar de un Array:
    
import numpy as np
np.set_printoptions(threshold=np.inf)

"""


# Importing the libraries

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Importamos el datasett de entrenamiento de Titanic train.csv
# La opcion: index_col = 0, indica que utilizarà la primera columna del csv como indice.
dataset = pd.read_csv('train.csv', index_col=0)

# Por si acaso lo necesitamos mas tarde, guardamos el dataset otriginal
dataset_original = dataset


# determine categorical and numerical features
numerical_ix = dataset.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = dataset.select_dtypes(include=['object', 'bool']).columns

print("Valores Numericos:   ", numerical_ix.values)
print("Valores Categoricos: ", categorical_ix.values)

# Rellenaremos el campo edad con la media de edad de todo el barco
dataset['Age'] = dataset['Age'].fillna(int(dataset_original['Age'].mean()))


data = dataset[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]


# Comprobamos si existen valores Nan en algun campo
ColumnsNan = data.isnull().sum()

"""
Separamos los valores entre Variables independientes (X)
, y Variable dependiente (y) o variable a predecir
"""
X = data.iloc[:, 1:].values
y = data.iloc[:, :1].values


# Transformamos las columnas categoricas en valores numericos
transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",                   # Un nombre de la transformación
         OneHotEncoder(categories='auto'),    # La clase a la que transformar
         [1]                                  # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
#  https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html


# Dividir entre entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Escalado de variables (Estandarización)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
