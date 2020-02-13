# -*- coding: utf-8 -*-
import io
import os
import time

import cv2
import numpy as np
from flask import Flask, request, redirect, url_for,jsonify,render_template
from flask_cors import CORS

from detector import Detector
from recoer import Recoer

detector = Detector('./data/models/ctpn.pb')
recoer = Recoer('./tf_crnn/data/chars/chn.txt', './data/models/crnn.pb')

app = Flask(__name__)
app.results = []
app.config['UPLOAD_FOLDER'] = 'examples/uploads'
app.config["CACHE_TYPE"] = "null"
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024    # 1 Mb limit
app.config['idcard.img'] = app.config['UPLOAD_FOLDER'] + "/idcard.img"
app.filename = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def get_cv_img(r):
    f = r.files['img']
    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    nparr = np.fromstring(in_memory_file.getvalue(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def process(img):
    start_time = time.time()
    rois = detector.detect(img)
    print("CTPN time: %.03fs" % (time.time() - start_time))

    start_time = time.time()
    ocr_result = recoer.recognize(rois, img)
    print("CRNN time: %.03fs" % (time.time() - start_time))

    sorted_data = sorted(zip(rois, ocr_result), key=lambda x: x[0][1] + x[0][3] / 2)
    rois, ocr_result = zip(*sorted_data)

    res = {"results": []}

    for i in range(len(rois)):
        res["results"].append({
            'position': rois[i],
            'text': ocr_result[i]
        })

    return res

import json

@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        img = get_cv_img(request)
        ret = process(img)
        return json.dumps(ret)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
