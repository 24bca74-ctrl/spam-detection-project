from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])

def home():

    prediction=""

    if request.method=="POST":

        message = request.form["message"]

        data = vectorizer.transform([message])

        result = model.predict(data)

        if result[0]==1:

            prediction="Spam Message"

        else:

            prediction="Normal Message"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__=="__main__":

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)