from flask import Flask, render_template, request, flash, session
from flask_session import Session
import os
from PIL import Image
from utils import *

app = Flask(__name__)
sess = Session()
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']

@app.route("/", methods=['GET','POST'])
def home():

    if request.method =="POST":
        inputfile = request.files['file']
        if inputfile.filename!='':
            inputfile = request.files['file']
            inputfile.save(inputfile.filename)

            im = Image.open(inputfile.filename)
            im.save('input.png')

            print('Processing image ...\n')
            foot_width, foot_length = full_pipeline('input.png')
            try:
                shoes_size = get_size(foot_width, foot_length)
            except:
                shoes_size = False

            if shoes_size:
                flash('Recommended shoes size (Men - VN): '+str(shoes_size))
            else:
                flash('Something not right. Make sure the image has all 4 corners of the paper.')
            flash('Measured foot width: '+str(round(foot_width,2))+'cm - foot length: '+str(round(foot_length,2)) +' cm.')
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)
    app.run(debug=True)
