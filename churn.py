import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    churn = pd.read_csv('Churn_Modelling.csv')
    churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    churn = pd.get_dummies(churn, columns=['Geography', 'Gender'], drop_first=True)
    for col in churn.columns:
        if churn[col].dtype == 'bool':
            churn[col] = churn[col].astype(int)
    return churn

churn = load_data()

# Feature selection
X = churn.drop(columns='Exited')
Y = churn['Exited']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Data scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model definition
def create_model():
    input_layer = tf.keras.layers.Input(shape=(X.shape[1],))
    dense_layer_1 = tf.keras.layers.Dense(11, activation='relu')(input_layer)
    dense_layer_2 = tf.keras.layers.Dense(11, activation='relu')(dense_layer_1)
    dense_layer_3 = tf.keras.layers.Dense(8, activation='relu')(dense_layer_2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer_3)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = create_model()

# Training the model
history = model.fit(x_train_scaled, y_train, batch_size=50, epochs=50, verbose=1, validation_split=0.2)

# Streamlit app
st.title('Churn Prediction App')

st.sidebar.header('User Input Features')

# User input
def user_input_features():
    CreditScore = st.sidebar.slider('CreditScore', 350, 850, 650)
    Geography = st.sidebar.selectbox('Geography', ['France', 'Spain', 'Germany'])
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Age = st.sidebar.slider('Age', 18, 92, 35)
    Tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance = st.sidebar.slider('Balance', 0.0, 250898.09, 100000.0)
    NumOfProducts = st.sidebar.slider('NumOfProducts', 1, 4, 2)
    HasCrCard = st.sidebar.selectbox('HasCrCard', [0, 1])
    IsActiveMember = st.sidebar.selectbox('IsActiveMember', [0, 1])
    EstimatedSalary = st.sidebar.slider('EstimatedSalary', 0.0, 200000.0, 100000.0)

    data = {'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocessing user input
input_df = pd.get_dummies(input_df, columns=['Geography', 'Gender'], drop_first=True)

# Align the input with training data
missing_cols = set(X.columns) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0
input_df = input_df[X.columns]

input_df_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_df_scaled)
prediction_prob = prediction[0][0]

st.header('Prediction Result')
if prediction_prob >= 0.5:
    st.write('The customer is predicted **to leave** the bank.')
else:
    st.write('The customer is predicted **not to leave** the bank.')

st.write(f'Probability of churn: {prediction_prob:.2f}')

# Visualizations
st.header('Model Training History')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'], label='Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Loss Over Epochs')
ax1.legend()

ax2.plot(history.history['acc'], label='Accuracy')
ax2.plot(history.history['val_acc'], label='Val Accuracy')
ax2.set_title('Accuracy Over Epochs')
ax2.legend()

st.pyplot(fig)
