from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from mcq_generator import generate_mcqs
app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r'/*': {
        'origins': '*'
    }
})
@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/gen', methods=['POST'])
def generate():
    text = request.json['text']
    if len(text) == 0:
        return jsonify(['enter some text to generate mcqs'])
    response = jsonify(generate_mcqs(text))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8001)
