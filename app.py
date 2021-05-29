import numpy as np
import pickle
import pandas as pd
from flask import Flask,request,render_template

app=Flask(__name__)


pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/home')
def home():
     return render_template("home.html")


@app.route('/predict',methods=["POST"])
def predict():
    
     variance=request.form["variance"]
     skewness=request.form["skewness"]
     curtosis=request.form["curtosis"]
     entropy=request.form["entropy"]
     
     prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
          

     output=prediction[0]

     if output==0:

          return render_template('home.html',prediction_text="The note is not authentic")
     else:
          return render_template('home.html',prediction_text="The note is authentic")


     return render_template("home.html")
 


if __name__=="__main__":
    app.run(debug=True)