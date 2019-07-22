from flask import Flask, render_template, request

from model.isParkinson import ip1

from werkzeug import secure_filename
app = Flask(__name__)


UPLOAD_FOLDER = '/model/for_eval'
@app.route('/')
def load_webpage():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def upload_image():
    if request.method.lower() == 'post':
        file = request.files['fileToUpload']

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        variable = ip1(filename)
        return render_template('results.html', value=)


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000