import cv2
import streamlit as st
from PIL import Image
import numpy as np

st.sidebar.title("Face Trimmer")
def read_model():
    face_cascade_path = './haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return face_cascade

file = st.sidebar.file_uploader("File uploader",type=["jpg","png","jpeg"])

if not file:
    st.warning('Please input a image')
    st.stop()

src=Image.open(file)
before,after = st.columns(2)

with before:
    st.subheader("Before")
    st.image(src)

with after:
    with st.spinner("Now Trimming..."):

        src = np.array(src)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        face_cascade = read_model()
        faces = face_cascade.detectMultiScale(src_gray)

        if len(faces) == 0:
            st.warning("顔が検出されませんでした")
            st.stop()
        cv2.imwrite("after.jpg",src[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]])
        src=Image.open("after.jpg")
        st.subheader("After")
        st.image(src)

        with open("after.jpg", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="after.jpg",
                    mime="image/jpg"
                )