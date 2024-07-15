import numpy as np
import pickle
from flask import Flask, request, render_template


app = Flask(__name__, template_folder='templates')
pickle_file_path = r'model.pkl'


# Load the model from the pickle file
with open(pickle_file_path, 'rb') as file:
    try:
        model = pickle.load(file)
    except ValueError as e:
        if "itemsize" in str(e):
            # Handle the incompatible dtype issue
            msg = "Incompatible dtype issue in the node array. Try retraining and saving the model with the latest scikit-learn version."
            raise ValueError(msg)
        else:
            raise e

@app.route('/')  # Rendering the HTML templates for the home page
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])  # Rendering the HTML templates for the prediction page
def predict():
    return render_template("inner-page.html")

@app.route('/submit', methods=["POST", "GET"])  # Route to show the predictions in a web UI
def submit():
    # Reading the inputs given by the user
    input_feature = [int(float(x)) for x in request.form.values()]
    input_feature = [np.array(input_feature)]

    # Predictions using the loaded model file
    print(type(model))
    prediction = model.predict(input_feature)
    prediction = int(prediction)

    if prediction == 0:
        return render_template("output.html", result="The apple is of bad quality, with its characterizations ")
    else:
        return render_template("output.html", result="The apple is of good quality, with its characterizations")

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=2000)



