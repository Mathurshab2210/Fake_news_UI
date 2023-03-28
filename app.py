from flask import Flask, escape, request, render_template
import pickle

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)

        predict = model.predict(vector.transform([news]))
        if predict == 0:
            n="News is real"
        else:
            n="News is fake"

        return render_template("prediction.html", prediction_text="Given -> {}".format(n))


    else:
        return render_template("prediction.html")


if __name__ == '__main__':
     app.debug = True
     app.run()
