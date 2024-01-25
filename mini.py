from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import numpy
import os


app = Flask(__name__)

# Load the model and tokenizer

# Define the maximum sequence length


# Define the labels
labels = ['Negative', 'Positive']

# Define the route for predicting the sentiment


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    # Get the text from the request
    text = request.json['text']

    # Convert the text to a sequence of integers
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the sequence
    sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Get the prediction
    prediction = model.predict(sequence)

    # Get the label with the highest probability
    label = labels[np.argmax(prediction)]

    # Return the label as a JSON response
    return jsonify({'sentiment': label})


if __name__ == '__main__':
    app.run(debug=True)
