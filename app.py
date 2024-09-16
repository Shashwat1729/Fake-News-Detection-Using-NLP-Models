from flask import Flask, request, render_template
from model import predict_fake_news
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['news_text']
        prediction = predict_fake_news(text)
        return render_template('index.html', prediction=prediction, news_text=text)

if __name__ == '__main__':
    app.run(debug=True)
