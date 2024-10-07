from flask import Flask, render_template, jsonify, request
from model import predict_image
from markupsafe import Markup
import utils

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return render_template('index.html', status=400, res="No file uploaded.")
            
            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            res = Markup(utils.disease_dic[prediction])
            return render_template('display.html', status=200, result=res)
        
        except Exception as e:
            print(f"Error during prediction: {e}")  # Log the error
            return render_template('index.html', status=500, res="Internal Server Error")
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
