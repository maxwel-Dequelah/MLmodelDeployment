import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import string
import joblib  # Import joblib for pickle model saving


# Step 1: Generate the Dataset with SQL Injection Patterns
def generate_data(num_rows=1000):
  # SQL injection patterns
  sql_injection_patterns = [
      "SELECT * FROM users WHERE id=",
      "' OR 1=1 --",
      "DROP TABLE",
      "--",
      ";",
      "'",
      "UNION SELECT",
      "1' OR 'a'='a",
      "DROP DATABASE",
      "UPDATE users SET password=",
      "INSERT INTO users VALUES",
      "';--",
      "' UNION ALL SELECT",
      "SELECT password FROM users WHERE username=",
      "EXCEPT SELECT",
      "ORDER BY"
  ]

  # Generate the dataset
  data = []
  for _ in range(num_rows):
    is_malicious = random.random() < 0.02  # 2% malicious
    if is_malicious:
      account_number = random.choice(sql_injection_patterns) + str(random.randint(1000, 9999))
      amount = random.choice(sql_injection_patterns) + str(random.randint(1, 10000))
    else:
      account_number = ''.join(random.choices(string.digits, k=10))
      amount = str(random.randint(1, 10000))

    transaction_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    data.append({
        'Transaction ID': transaction_id,
        'Account Number': account_number,
        'Amount': amount,
        'Label': 1 if is_malicious else 0
    })

  return pd.DataFrame(data)


# Step 2: Define and Train the CNN Model
def train_cnn_model(df):
  tokenizer = Tokenizer(char_level=True)
  tokenizer.fit_on_texts(df['input'])

  max_len = 50  # Max sequence length
  sequences = tokenizer.texts_to_sequences(df['input'])
  X = pad_sequences(sequences, maxlen=max_len)
  y = df['Label'].values

  # Create the CNN model
  cnn_model = tf.keras.Sequential([
      tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
      tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01)),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Dropout(0.5),  # Increase dropout to avoid overfitting
      tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
  ])

  cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Train the model
  cnn_model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2, class_weight={0: 1., 1: 15.})

  return cnn_model, tokenizer, max_len


# Step 3: Save the Model to Disk (as a pickle file)
# Step 3: Save the Model to Disk (as a pickle file)
def save_model(cnn_model, tokenizer, max_len):
  # Save the model and other components using joblib
  model_data = {
      'model': cnn_model,
      'tokenizer': tokenizer,
      'max_len': max_len
  }
  joblib.dump(model_data, 'sql_injection_detection_model.pkl')
  print("Model saved as sql_injection_detection_model.pkl")


# Step 4: Load the Model and Predict Malicious Transactions
def load_and_predict_model(model_path, sample_data):
  # Load the saved model data
  loaded_model_data = joblib.load(model_path)
  cnn_model = loaded_model_data['model']
  tokenizer = loaded_model_data['tokenizer']
  max_len = loaded_model_data['max_len']

  print("Model loaded successfully!")

  sequences = tokenizer.texts_to_sequences([entry['input'] for entry in sample_data])
  X_sample = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

  cnn_predictions = (cnn_model.predict(X_sample) > 0.5).astype('int32')

  for idx, entry in enumerate(sample_data):
      print(f"Transaction ID: {entry['Transaction ID']}")
      print(f"Account Number: {entry['Account Number']}")
      print(f"Amount: {entry['Amount']}")
      print(f"Prediction (CNN): {'Malicious' if cnn_predictions[idx] == 1 else 'Not Malicious'}")
      print("-" * 50)

# Example Usage
if __name__ == "__main__":
 # Step 1: Generate Data
    df = generate_data(5000)  # Adjust number of rows as needed
    df['input'] = df['Account Number'] + ' ' + df['Amount'].astype(str)  # Combine account number and amount

    # Step 2: Train CNN Model
    cnn_model, tokenizer, max_len = train_cnn_model(df)

    # Step 3: Save the Model
    save_model(cnn_model)

    # Step 4: Test with Sample Data
    sample_data = [
        {"Transaction ID": "TX12345A", "Account Number": "1234567890", "Amount": "1000", "input": "1234567890 1000"},
        {"Transaction ID": "TX12346B", "Account Number": "' OR 1=1 --", "Amount": "5000", "input": "' OR 1=1 -- 5000"},
        {"Transaction ID": "TX12347C", "Account Number": "9876543210", "Amount": "2000", "input": "9876543210 2000"},
        {"Transaction ID": "TX12348D", "Account Number": "SELECT * FROM users WHERE id=", "Amount": "3000", "input": "SELECT * FROM users WHERE id= 3000"},
        {"Transaction ID": "TX12349E", "Account Number": "1122334455", "Amount": "1500", "input": "1122334455 1500"},
        {"Transaction ID": "TX12350F", "Account Number": "'; DROP TABLE users;", "Amount": "20000", "input": "'; DROP TABLE users; 20000"},
        {"Transaction ID": "TX12351G", "Account Number": "9988776655", "Amount": "7500", "input": "9988776655 7500"},
        {"Transaction ID": "TX12352H", "Account Number": "UNION SELECT", "Amount": "12000", "input": "UNION SELECT 12000"},
        {"Transaction ID": "TX12353I", "Account Number": "1029384756", "Amount": "1300", "input": "1029384756 1300"},
        {"Transaction ID": "TX12354J", "Account Number": "1122334455", "Amount": "4000", "input": "1122334455 4000"}
    ]

    # Step 5: Test the Model on Sample Data
    load_and_predict_model('sql_injection_detection_model.h5', tokenizer, max_len, sample_data)