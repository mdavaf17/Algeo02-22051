import os
from flask import Flask, request, redirect, url_for, render_template, send_file
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
from fpdf import FPDF
from datetime import datetime


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


def report_pdf(query_path, results, duration, method):
	pdf = FPDF('P', 'mm', 'A4')
	pdf.set_margins(left=10, top=5, right=10)
	pdf.add_page()
	pdf.set_font('Times', '', 18)
	pdf.cell(w=200, h=10, txt='OpenEigen Image Retrival Result', ln=1, align='C')
	pdf.ln(10)
	pdf.set_font('Times', '', 12)
	now = datetime.now()
	dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
	pdf.cell(w=200, h=5, txt="Date: " + dt_string, ln=1, align='R')
	pdf.ln(5)

	pdf.cell(w=100, h=5, txt="Image Query:", ln=1)
	pdf.image(name=query_path, x=10, y=40, h=50)
	pdf.ln(55)

	pdf.cell(w=100, h=5, txt="Results:", ln=1)
	pdf.cell(w=100, h=5, txt= f"{str(len(results))} images found with similarity > 60% in {str('{:.3f}'.format(duration))} seconds using {'COLOR Feature' if method else 'TEXTURE Feature'}", ln=1)

	imgNumber = list(results.keys())
	imgPercentage = list(results.values())

	print(results)
	print(imgNumber)
	print(imgPercentage)

	firstPageImg = imgNumber[:9]

	absis=10; ordinat=110
	for i in range(len(firstPageImg)):
		pdf.image(name=f"../test/dataset/{imgNumber[i]}.jpg", x=absis, y=ordinat, w=50, h=50)
		pdf.text(x=absis+20, y=ordinat+54, txt=str("{:.3f}".format(imgPercentage[i]))+"%")
		absis += 60

		i += 1

		if i % 3 == 0:
			absis = 10
			ordinat += 60
	

	if len(imgNumber) > 9:
		idxSisa = 9
		sisa = imgNumber[idxSisa:]

		while len(sisa) != 0:
			pdf.add_page()
			absis=10; ordinat=20
			idx = 0
			if len(sisa) >= 13:
				idx = 12
			else:
				idx = len(sisa) % 13

			for i in range(idx):
				pdf.image(name=f"../test/dataset/{imgNumber[i+idxSisa]}.jpg", x=absis, y=ordinat, w=50, h=50)
				pdf.text(x=absis+20, y=ordinat+54, txt=str("{:.3f}".format(imgPercentage[i+idxSisa]))+"%")
				absis += 60

				i += 1

				if i % 3 == 0:
					absis = 10
					ordinat += 60
			idxSisa += 12
			sisa = imgNumber[idxSisa:]

	filename = os.path.basename(query_path)
	number = filename.split('.')[0]
	pdf.output(f'../test/report/report_{number}.pdf', 'F')


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

				c, h, e, d, asm, en, co, ic, dt = contrast(symmetric_matrix_normalized), homogeneity(symmetric_matrix_normalized), entropy(symmetric_matrix_normalized), dissimilarity(symmetric_matrix_normalized), asm_val(symmetric_matrix_normalized), math.sqrt(asm_val(symmetric_matrix_normalized)), correlation(symmetric_matrix_normalized), inverse_contrast(symmetric_matrix_normalized), distortion(symmetric_matrix_normalized)
				res = search_texture(c, h, e, d, asm, en, co, ic, dt)
				isTexture = 1
			else:
				img = cv2.imread(os.path.join(app.config['UPLOAD_CITRA'], next_filename))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				payload = build_vector(split_image(hsv_quantify(rgb_to_hsv(img))))

				res = search_color(payload)

			image_data = get_image_data()
			endT = time.time()
			duration = endT - startT

			report_pdf(os.path.join(app.config['UPLOAD_CITRA'], next_filename), res, duration, isTexture)
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


@app.route('/report/', methods=['GET'])
def download():
    number = request.args.get('number')  # Get the 'number' query parameter from the URL
    if number is not None:
        path = os.path.join(app.config['UPLOAD_DIR'], "report", f"report_{number}.pdf")
        return send_file(path, as_attachment=True)
    else:
        return "Number parameter missing or invalid"


if __name__ == "__main__":
    app.run(debug=True)