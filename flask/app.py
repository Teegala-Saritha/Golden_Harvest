import numpy as np
import pickle
import pandas
from flask import Flask, request, render_template


app = Flask(__name__)
model = pickle.load(open(r'model.pkl', 'rb'))
# scale = pickle.load(open(r'scale1.pkl','rb'))

@app.route('/') # rendering the html template
def home():
    return render_template('home.html')
@app.route('/about') # rendering the html template
def about():
    return render_template('about.html')


@app.route('/predict',methods=["POST","GET"]) # rendering the html template

def predict() :
    return render_template("home.html")

@app.route('/submit',methods=["POST","GET"])# route to show the predictions in a web UI
def submit():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values() ]  
    #input_feature = np.transpose(input_feature)
    input_feature=[np.array(input_feature)]
    print(input_feature)
    names = ['A_id','Size','Weight','Sweetness','Crunchiness','Juciness','Ripeness','Acidity']
    data = pandas.DataFrame(input_feature,columns=names)
    print(data)

    # predictions using the loaded model file
    prediction=model.predict(data)

    #prediction = int(prediction)
    print(type(prediction))
    
    if prediction==0:
          return render_template("status.html", result="The apple is of bad quality,with its characterizations")
    else:
          return render_template("status.html", result="The apple is of good quality,with its characterizations")
   
     # showing the prediction results in a UI
     
if __name__=="__main__":
     app.run( debug=False, port = 4000)    # running the app
