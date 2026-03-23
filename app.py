
from flask import Flask, render_template, request, jsonify
import io, time, os, base64
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_FILENAME = "asl_10class_model.h5"
model = load_model(MODEL_FILENAME)
IMG_SIZE = model.input_shape[1]
CLASSES = ["A","B","C","D","E","F","G","H","I","J"]

def apply_edge_mode(img_rgb, mode):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mode=="none":
        out=bgr
    elif mode=="canny":
        e=cv2.Canny(gray,50,150); out=cv2.cvtColor(e,cv2.COLOR_GRAY2BGR)
    elif mode=="sobel":
        sx=cv2.Sobel(gray,cv2.CV_64F,1,0); sy=cv2.Sobel(gray,cv2.CV_64F,0,1)
        sob=np.uint8(np.clip(np.sqrt(sx*sx+sy*sy),0,255)); out=cv2.cvtColor(sob,cv2.COLOR_GRAY2BGR)
    elif mode=="laplacian":
        l=np.uint8(np.clip(abs(cv2.Laplacian(gray,cv2.CV_64F)),0,255)); out=cv2.cvtColor(l,cv2.COLOR_GRAY2BGR)
    else:
        out=bgr
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data=request.get_json()
    img_b64=data.get("image")
    mode=data.get("edge_mode", "none")
    if img_b64 is None:
        return jsonify({"error":"no image"}), 400
    # decode image
    header, encoded = img_b64.split(",",1) if "," in img_b64 else ("", img_b64)
    img_bytes = base64.b64decode(encoded)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(pil)
    start=time.time()
    proc = apply_edge_mode(img, mode)
    # prepare processed image as base64 PNG to send back for display
    proc_pil = Image.fromarray(proc)
    buf = io.BytesIO()
    proc_pil.save(buf, format="PNG")
    proc_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    # prediction
    proc_resized = cv2.resize(proc, (IMG_SIZE, IMG_SIZE)).astype("float32")/255.0
    pred = model.predict(np.expand_dims(proc_resized,0))[0]
    idx = int(np.argmax(pred))
    label = CLASSES[idx]
    conf = float(pred[idx] * 100.0)
    end=time.time()
    t = end - start
    fps = 1.0 / t if t>0 else 0.0
    return jsonify({
        "pred": label,
        "confidence": round(conf,2),
        "fps": round(fps,2),
        "proc_image": "data:image/png;base64," + proc_b64
    })

if __name__=="__main__":
    import webbrowser, threading
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
