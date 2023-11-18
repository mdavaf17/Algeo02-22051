import os
from flask import Flask, flash, request, redirect, url_for, render_template
from color import *
from texture import *
from PIL import Image
from werkzeug.utils import secure_filename
import time

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_CITRA = '../test/citra/'
UPLOAD_DIR = '../test/'

app = Flask(__name__, static_folder='../test')
app.secret_key = "secret key"
app.config['UPLOAD_CITRA'] = UPLOAD_CITRA
app.config['UPLOAD_DIR'] = UPLOAD_DIR 

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_next_filename(upload_folder):
    files = os.listdir(upload_folder)
    file_numbers = []

    for file in files:
        name, ext = os.path.splitext(file)
        if ext[1:] in ALLOWED_EXTENSIONS:
            try:
                file_numbers.append(int(name))
            except ValueError:
                pass

    if not file_numbers:
        return '0'
    else:
        return str(max(file_numbers) + 1)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
	if 'file' not in request.files:
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		next_filename = get_next_filename(UPLOAD_CITRA)
		next_filename += os.path.splitext(filename)[1]
		file.save(os.path.join(UPLOAD_CITRA, f"{next_filename}"))
		# file.save(os.path.join(app.config['UPLOAD_CITRA'], filename))
		startT = time.time()
		res = ""
		if request.form.get('texture'):
			img = Image.open(os.path.join(app.config['UPLOAD_CITRA'], next_filename))
			framework_matrix = create_framework_matrix(np.array(preprocess_image(img)))
			symmetric_matrix = framework_matrix + framework_matrix.transpose()
			symmetric_matrix_normalized = symmetric_matrix / symmetric_matrix.sum()

			c, h, e = contrast(symmetric_matrix_normalized), homogeneity(symmetric_matrix_normalized), entropy(symmetric_matrix_normalized)
			res = search_texture(c, h, e)
		else:
			img = cv2.imread(os.path.join(app.config['UPLOAD_CITRA'], next_filename))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			payload = build_vector(split_image(hsv_quantify(rgb_to_hsv(img))))

			res = search_color(payload)
		endT = time.time()
		duration = endT - startT

		return render_template('index.html', filename=next_filename, result=res, duration=duration)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)


@app.route('/upload/',methods = ['GET','POST'])
def upload_dataset():
	if request.method =='POST':
		files = request.files.getlist("file[]")
		infoDir = "success"
		i = 0
		try:
			for index, file in enumerate(files, start=1):
				# path = os.path.dirname(file.filename) Nama dir original saat diupload
				path = "client-dataset"
				path2 = os.path.join(app.config['UPLOAD_DIR'], path)
				if not os.path.exists(path2):
					os.mkdir(path2)

				original_filename, original_extension = os.path.splitext(file.filename)
				new_filename = f"{index}{original_extension}"
				full_path = os.path.join(app.config['UPLOAD_DIR'], path, new_filename)
				file.save(full_path)
		except Exception as e:
			infoDir = "something went wrong"

	return render_template('index.html', infoDir=infoDir)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='citra/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)