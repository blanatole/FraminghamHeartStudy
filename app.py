import sklearn
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model đã lưu (nếu bạn đã lưu model sau khi huấn luyện)
model = pickle.load(open('framingham_heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Tạo giao diện
st.title('Dự đoán nguy cơ bệnh tim')

# Nhập dữ liệu từ người dùng
sex = st.selectbox('Giới tính (1 = Nam, 0 = Nữ)', [0, 1])
age = st.slider('Tuổi', 20, 80, 50)
education = st.selectbox('Trình độ học vấn', [1, 2, 3, 4])
currentSmoker = st.selectbox('Hút thuốc lá hiện tại (1 = Có, 0 = Không)', [0, 1])
cigsPerDay = st.number_input('Số điếu thuốc mỗi ngày', min_value=0, max_value=100, value=10)
BPmeds = st.selectbox('Sử dụng thuốc hạ huyết áp (1 = Có, 0 = Không)', [0, 1])
prevalentStroke = st.selectbox('Đã từng bị đột quỵ (1 = Có, 0 = Không)', [0, 1])
prevalentHyp = st.selectbox('Huyết áp cao (1 = Có, 0 = Không)', [0, 1])
diabetes = st.selectbox('Bị tiểu đường (1 = Có, 0 = Không)', [0, 1])
totChol = st.number_input('Cholesterol tổng (mg/dL)', min_value=100, max_value=400, value=200)
sysBP = st.number_input('Huyết áp tâm thu (mmHg)', min_value=90, max_value=200, value=120)
diaBP = st.number_input('Huyết áp tâm trương (mmHg)', min_value=60, max_value=130, value=80)
BMI = st.number_input('Chỉ số BMI', min_value=10.0, max_value=50.0, value=25.0)
heartRate = st.number_input('Nhịp tim (lần/phút)', min_value=40, max_value=150, value=70)
glucose = st.number_input('Mức đường huyết (mg/dL)', min_value=50, max_value=200, value=100)

cigsPerDay = np.log1p(cigsPerDay)
totChol = np.log1p(totChol)
sysBP = np.log1p(sysBP)
diaBP = np.log1p(diaBP)
BMI = np.log1p(BMI)
heartRate = np.log1p(heartRate)
glucose = np.log1p(glucose)
age = np.log1p(age)


# Chuyển dữ liệu thành array
input_data = np.array([[sex, age, education, currentSmoker, cigsPerDay, BPmeds, prevalentStroke, 
                        prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])


input_data_scaled = scaler.transform(input_data)

# Dự đoán
if st.button('Dự đoán'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.write(f'Xác suất không mắc bệnh tim: {(prediction_proba[0][0]*100):.2f}%')
    st.write(f'Xác xuất mắc bệnh tim: {(prediction_proba[0][1]*100):.2f}%')

    st.subheader('Kết luận:')
    if prediction[0] == 1:
        st.write('Nguy cơ cao mắc bệnh tim.')
    else:
        st.write('Nguy cơ thấp mắc bệnh tim.')

    
    
