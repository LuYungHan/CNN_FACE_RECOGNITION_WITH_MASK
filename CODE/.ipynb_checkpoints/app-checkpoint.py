from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file
from PIL import Image
from backbones import get_model
from Final_realtime_face_recognition import inference
import torch
import numpy as np
import json
import traceback
import time
from datetime import datetime,timedelta 
# Ttime = str(datetime.now())
# print(Ttime)


net = get_model('r50', fp16=False)
net.load_state_dict(torch.load('./work_dirs/ms1mv2_r50/model.pt'))
net.eval()

app = Flask(__name__)

@app.route("/face_recognition", methods=["POST"])
def face_recognition():
    t_start = time.time()
    try:
        file = request.files["file"]
        position = json.loads(request.form.get("position"))

        img = Image.open(file)
        img_array = np.array(img)
        img_array = img_array[
            position["top"] : position["top"] + position["height"],
            position["left"] : position["left"] + position["width"],
        ]
        
        im = Image.fromarray(img_array)

        # face recognition
        PPS_CODE = inference(net, img_array, './all_face_feat_0824.csv')

        response = {
            "state": 0,
            "identity": PPS_CODE,
            "description": "success",
        }

        return jsonify(response)
    except:
        traceback.print_exc()
        response = {"state": -1, "description": traceback.format_exc()}
        return jsonify(response)


if __name__ == "__main__":
    Ttime = str(datetime.now())
    print(Ttime)
    app.debug = False
    app.run(host="0.0.0.0", port=50009)
    Ttime = str(datetime.now())
    print(Ttime)