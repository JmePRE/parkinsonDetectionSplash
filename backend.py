from flask import Flask, render_template, request

import model.isParkinson
import os, time
from werkzeug import secure_filename
import tensorflow as tf
app = Flask(__name__, )


UPLOAD_FOLDER = 'model\\for_eval'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
nn = model.isParkinson.Neural()
@app.route('/')
def load_webpage():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def upload_image():
    if request.method.lower() == 'post':
        file = request.files['fileToUpload']

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('fn'+filename)
        variable = nn.ip1('model\\for_eval\\'+filename)
        return render_template('results.html', value=variable, value2=variable)

@app.route('/login.php')
def login():
    return render_template('login.php')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/results.html')
def results():
    return render_template('results.html')

@app.route('/camera.html')
def camera():
    return render_template('camera.html')

@app.route('/imagelog.html')
def imagelog():
    return render_template('imagelog.html')
    
@app.route('/image')
def image_display():
    if (request.method.lower() == 'post':
        file = request.form['fileName']
        
        variable = nn.ip1('model\\for_eval\\'+file)
        
        
    return render_template('result.html', value='model\\for_eval\\'+file, value2=variable)
        
@app.route('/parkinsons.html')
def parkinsons():
    return render_template('parkinsons.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0') #run app in debug mode on port 5000
