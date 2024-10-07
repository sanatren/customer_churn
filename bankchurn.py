import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

churn = pd.read_csv('Churn_Modelling.csv')
print(churn.info())

pd.set_option('display.max_columns', None)

churn.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)

print(churn.isnull().sum())
print(churn.duplicated().sum())

print(churn['Exited'].value_counts())
print(churn['Geography'].value_counts())

#hot encoding
churn = pd.get_dummies(churn,columns=['Geography','Gender'],drop_first=True)

for col in churn.columns:
    if churn[col].dtype == 'bool':
        churn[col] = churn[col].astype(int)

print(churn.head())

X = churn.drop(columns = 'Exited')
Y = churn['Exited']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


import tensorflow as tf

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model

input_layer = Input(shape=(X.shape[1],))  # Corrected X.shape[1]
dense_layer_1 = Dense(11, activation='relu')(input_layer)
dense_layer_2 = Dense(11, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(8, activation='relu')(dense_layer_2)
output = Dense(1, activation='sigmoid')(dense_layer_3)  # Corrected Y.shape[1]

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

history = model.fit(x_train_scaled, y_train, batch_size=50, epochs=50, verbose=1, validation_split=0.2)

y_pred = model.predict(x_test_scaled)
y_pred_class = np.round(y_pred).astype(int)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print("Accuracy score:", score)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()