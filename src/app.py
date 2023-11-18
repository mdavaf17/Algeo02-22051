import os
from flask import Flask, request, redirect, url_for, render_template
from color import *
from texture import *
from PIL import Image
from werkzeug.utils import secure_filename
import time
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import shutil


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_CITRA = '../test/citra/'
UPLOAD_DIR = '../test/'

app = Flask(__name__, static_folder='../test')
app.secret_key = "secret key"
app.config['UPLOAD_CITRA'] = UPLOAD_CITRA
app.config['UPLOAD_DIR'] = UPLOAD_DIR
executor = ThreadPoolExecutor(3)


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


def get_image_data():
    directory = "../test/dataset/"
    images_data = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            images_data.append(filename)
    return images_data


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
	if os.path.exists("../test/db_color.csv") and os.path.exists("../test/db_texture.csv"):
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
			startT = time.time()
			res = ""
			isTexture = ""

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
				isTexture = 1

			image_data = get_image_data()
			endT = time.time()
			duration = endT - startT

			return render_template('index.html', filename=next_filename, result=res, duration=duration, image_data=image_data, isTexture=isTexture)
	else:
		return render_template('index.html', infoDir="Database masih dalam Training atau Dataset belum diupload!")


@app.route('/upload/',methods = ['GET','POST'])
def upload_dataset():
	if request.method =='POST':
		files = request.files.getlist("file[]")
		infoDir = "Berhasil mengunggah dataset"
		try:
			for index, file in enumerate(files, start=0):
				path = "dataset"
				path2 = os.path.join(app.config['UPLOAD_DIR'], path)
				if not os.path.exists(path2):
					os.mkdir(path2)

				original_filename, original_extension = os.path.splitext(file.filename)
				if original_extension.replace('.', '') in ALLOWED_EXTENSIONS:
					new_filename = f"{index}{original_extension}"
					full_path = os.path.join(app.config['UPLOAD_DIR'], path, new_filename)
					file.save(full_path)
		
			executor.submit(save_color_csv)
			executor.submit(save_texture_csv)

		except Exception as e:
			infoDir = "Gagal menggunggah dataset"

	return render_template('index.html', infoDir=infoDir)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='citra/' + filename), code=301)


def is_valid_image_url(img_url):
    try:
        response = requests.head(img_url)
        return response.status_code == 200 
    except requests.RequestException:
        return False


def scrape_img(url):
	r = requests.get(url)
	soup = BeautifulSoup(r.text, 'html.parser')

	path = "dataset"
	path2 = os.path.join(app.config['UPLOAD_DIR'], path)
	if os.path.exists(path2):
		try:
			shutil.rmtree(path2)
		except OSError as e:
			print(f"Error: {e}")

	if not os.path.exists(path2):
		os.mkdir(path2)

	counter = 0

	try:	
		for img in soup.find_all('img'):
			img_url = img.get('src')
			if img_url:
				if not img_url.startswith(('http://', 'https://')):
					img_url = f"{url.rstrip('/')}/{img_url.lstrip('/')}"

			parsed_url = urlparse(img_url)
			image_extension = os.path.splitext(parsed_url.path)[1]
			if image_extension.replace('.', '') in ALLOWED_EXTENSIONS and is_valid_image_url(img_url):
				try:
					image_data = requests.get(img_url).content
					image_name = os.path.join(path2, f"{counter}{image_extension}")
					with open(image_name, 'wb') as f:
						print(counter, img_url)
						f.write(image_data)
					counter += 1
				except Exception as e:
					print(f"Failed to download {img_url}: {e}")
		
		executor.submit(save_color_csv)
		executor.submit(save_texture_csv)
	except:
		pass


@app.route('/scrape/', methods = ['POST'])
def job_scraper():
	if request.method == 'POST':
		url = request.form['site_url']
		executor.submit(scrape_img, url)
	
	return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)