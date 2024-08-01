from flask import Flask, request, jsonify
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Crop Prediction API"

@app.route("/predict", methods=['POST'])
def predict():
    # Getting form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Preparing the feature array
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Applying scalers
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    # Crop dictionary
    crop_dict = {1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas", 6: "Mothbeans", 7: "Mungbean",
                 8: "Blackgram", 9: "Lentil", 10: "Pomegranate", 11: "Banana", 12: "Mango", 13: "Grapes",
                 14: "Watermelon", 15: "Muskmelon", 16: "Apple", 17: "Orange", 18: "Papaya",
                 19: "Coconut", 20: "Cotton", 21: "Jute", 22: "Coffee"}

    # Returning JSON response
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return jsonify({"prediction": result})

# python main
if __name__ == "__main__":
    app.run(debug=True)
