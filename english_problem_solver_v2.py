# English Problem Solver v2
# 기능1. 그래프 이미지 업로드하면 gpt-4-vision이 답변
# - 그래프 이미지 업로드
# - 지문 텍스트 업로드

# 기능2. 텍스트 입력하면 gpt-4-turbo가 답변


from openai import OpenAI
from IPython.display import Image
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
# llm : langchain.ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import os
import base64
import requests

api_key = os.environ.get('OPENAI_API_KEY')


import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image as Img
from IPython.display import Image

# https://velog.io/@wonjun12/Streamlit-%ED%8C%8C%EC%9D%BC-%EC%97%85%EB%A1%9C%EB%93%9C


st.header("📄English Problem Solver📊")

# 사이드에 list의 선택박스를 생성한다.
select = st.sidebar.selectbox('메뉴', ['텍스트 분석', '그래프 이미지 분석'])

if select == '텍스트 분석':

    # 사용자 질문 입력
    question = st.text_area("문제를 입력해주세요", height=10)

    # 메시지 히스토리 생성
    messages=[{"role": "system", "content": """
               You are a helpful assistant. 
               Answer the questions given and explain your reasoning.
               You must answer in Korean."""}]

    question_dict = {
                "role": "user",
                "content": question ,  # 첫 번째 질문
            }

    # 메시지 히스토리에 질문 추가
    messages.append(question_dict)

    # 버튼 누르면 답변 받기 시작
    if st.button('풀어줘!'):
    # st.write('Why hello there')

        # 답변 받기
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",          # "gpt-3.5-turbo-0125", "gpt-4-0125-preview"
            messages = messages,
        )

        response = completion.choices[0].message.content

        st.info(response)


if select == '그래프 이미지 분석':
    # 파일 업로드 (업로드 기능)
    img_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

    # context(지문) 입력받기
    context = st.text_input("📄영어 지문을 입력해주세요", "")

    if (img_file is not None) and (len(context) != 0):    # 만약 그래프 이미지 파일과 지문 텍스트가 업로드되면,
        st.image(img_file, width=500)
        # image = Image.open(img_file)
        # print("image: ", image)
        st.text("Q. 다음 중 위 도표의 내용과 일치하지 않는 것은?")
        st.text_area("", context)

        # 버튼 누르면 답변 받기 시작
        if st.button('풀어줘!'):
            # --------------------------deplot으로 수치 추출하기-------------------------

            processor = Pix2StructProcessor.from_pretrained('nuua/ko-deplot')
            model = Pix2StructForConditionalGeneration.from_pretrained('nuua/ko-deplot')

            # 그래프 이미지에서 수치 추출
            image = Img.open(img_file)

            inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
            predictions = model.generate(**inputs, max_new_tokens=512)
            result = processor.decode(predictions[0], skip_special_tokens=True)

            st.text_area(result)

            # --------------------------gpt-4-vision으로 문제 풀기--------------------------

            # 메시지 히스토리 생성
            messages=[{"role": "system", "content": """
                       You must answer in You are a math assistant who does accurate calculations. 
                       Correct the algebra of the given numbers and choose the one you think is the most incorrect.
                       """}]


            # 입력 예시(image_path, prompt)
            image_path = img_file.name

            # 첫번째 질문 생성
            question = f""" 
            그래프 분석 result를 제공해줄게.
            result : {result}.

            Q. 다음 중 그래프 내용과 가장 틀린 문장이 몇 번인지 찾고, 그 이유를 설명해. 
            {context}
            """ 

            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
                    
            # Getting the base64 string
            base64_image = encode_image(image_path)

            question_dict = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                        },
                    ],
                    }

            # 메시지 히스토리에 질문 추가
            messages.append(question_dict)

            # 답변 받기
            completion = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages = messages,
            )

            response = completion.choices[0].message.content

            st.info(response)

    else:
        st.info('☝️ 그래프 이미지와 지문을 업로드하면 문제를 풀어드립니다.')


