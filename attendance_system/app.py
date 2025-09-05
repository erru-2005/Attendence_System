from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Placeholder endpoints for recognition, add, delete
@app.route('/recognize', methods=['POST'])
def recognize():
    # To be implemented: call your recognition logic
    return jsonify({'name': 'John Doe', 'time': '0.5s', 'accuracy': '99.9%'})

@app.route('/add_student', methods=['POST'])
def add_student():
    # To be implemented: call your add student logic
    return jsonify({'status': 'success'})

@app.route('/delete_student', methods=['POST'])
def delete_student():
    # To be implemented: call your delete student logic
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True) 