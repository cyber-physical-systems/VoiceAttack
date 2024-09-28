import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_shape, num_classes, lstm_units, dropout_rate, learning_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(data_path):
    data = pd.read_csv(data_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    X = data.drop('label', axis=1).values
    y = data['label'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=25)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    X_train_reshaped = np.reshape(X_train_pca, (X_train_pca.shape[0], 1, X_train_pca.shape[1]))
    X_test_reshaped = np.reshape(X_test_pca, (X_test_pca.shape[0], 1, X_test_pca.shape[1]))

    model = create_model((1, X_train_reshaped.shape[2]), y_train.shape[1], 256, 0.3, 0.001)
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=2, validation_data=(X_test_reshaped, y_test),
              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    _, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)

    predictions = model.predict(X_test_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    mcc = matthews_corrcoef(true_classes, predicted_classes)

    results = {
        'LSTM Units': 256,
        'Dropout Rate': 0.3,
        'Learning Rate': 0.001,
        'Accuracy': accuracy * 100,
        'MCC': mcc
    }
    return results


if __name__ == '__main__':
    data_path = ''

    results = train_and_evaluate(data_path)
    results_df = pd.DataFrame([results])  # Convert the dictionary to a DataFrame
    results_df.to_csv('LSTM.csv', index=False)
    print("Training results")
    print(results_df)

