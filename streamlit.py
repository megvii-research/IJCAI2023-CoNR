import cv2
import numpy as np
import streamlit as st
import os
import base64
import platform

st.set_page_config(layout="wide", page_title='CoNR demo', page_icon="🪐")

st.title('CoNR demo')
st.markdown(""" <style> 
            #MainMenu {visibility: hidden;} 
            footer {visibility: hidden;} 
            </style> """, unsafe_allow_html=True)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: 1920px 1080px;
#     background-attachment:fixed;
#     background-position:center;
#     background-repeat:no-repeat;
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# set_background('ipad_bg.png')

upload_img = (st.file_uploader("输入character sheet", "png", accept_multiple_files=True))

if st.button('RUN!'):
    if upload_img is not None:
        for i in range(len(upload_img)):
            with open('character_sheet/{}.png'.format(i), 'wb') as f:
                f.write(upload_img[i].read())

        st.info('努力推理中...')
        if platform.system() == 'Windows':
            os.system('infer.bat')
        elif platform.system() == 'Linux':
            os.system('sh infer.sh')
        else:
            raise NotImplementedError

        st.info('Done!')
        video_file=open('output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time=0)
    else:
        st.info('还没上传图片呢> <')
