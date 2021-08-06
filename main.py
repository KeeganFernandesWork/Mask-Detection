import streamlit as st
import torch
import cv2
import onnxruntime
import onnx
import numpy as np
from PIL import Image

ort_session = onnxruntime.InferenceSession("model_troch_export.onnx")
probs = torch.nn.Softmax()
image = st.file_uploader("Upload a Image")
def label_decode(x):
    if x == 1 :
        return "masks_correct"
    if x ==  0 :
        return "No_mask"
    if x == 2 :
        return "incorrectly_worn"

def get_prediction(img):
    img = cv2.resize(img, (256, 256))
    img = np.asarray(img, dtype = np.float32, order = "C").reshape(1, 3, 256, 256)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    logits = ort_session.run(None, ort_inputs)
    prob = probs(torch.tensor(logits).float().reshape(-1))
    top_p, top_class = prob.topk(1)
    print( prob,logits,top_p, top_class)
    return label_decode(int(top_class))
if st.button("Submit"):
    if image != None:
        st.image(image)
        st.header(get_prediction(np.array(Image.open(image))))
    else:
        st.header("Image not uploaded")