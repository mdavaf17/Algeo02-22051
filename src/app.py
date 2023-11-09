from flask import Flask, flash, request, redirect, url_for, render_template
import os
import urllib.request
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_CITRA = '../test/citra/'
UPLOAD_DIR = '../test/'

app = Flask(__name__, static_folder='../test')
app.secret_key = "secret key"
app.config['UPLOAD_CITRA'] = UPLOAD_CITRA
app.config['UPLOAD_DIR'] = UPLOAD_DIR 

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_CITRA'], filename))
		return render_template('index.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/upload/',methods = ['GET','POST'])
def upload_file():
	if request.method =='POST':
		files = request.files.getlist("file[]")
		infoDir = "success"
		try:
			for file in files:
				# path = os.path.dirname(file.filename) Nama dir original saat diupload
				path = "client-dataset"
				path2 = os.path.join(app.config['UPLOAD_DIR'], path)
				if not os.path.exists(path2):
					os.mkdir(path2)
				filename = os.path.join(path, secure_filename(os.path.basename(file.filename)))
				file.save(os.path.join(app.config['UPLOAD_DIR'], filename))
		except Exception as e:
			infoDir = "failed"

	return render_template('index.html', infoDir=infoDir)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='citra/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)