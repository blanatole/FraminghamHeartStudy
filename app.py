import sklearn
import streamlit as st
import pickle
import numpy as np
import pandas as pd

from logistic_regression_custom_balanced import LogisticRegressionCustomBalanced 

# Load model đã lưu (nếu bạn đã lưu model sau khi huấn luyện)
model = pickle.load(open('framingham_heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Tạo giao diện
st.title('Dự đoán nguy cơ bệnh tim')

# Nhập dữ liệu từ người dùng
sex = st.selectbox('Giới tính (1 = Nam, 0 = Nữ)', [0, 1]) 
age = st.slider('Tuổi', 20, 80, 50) 
currentSmoker = st.selectbox('Hút thuốc lá hiện tại (1 = Có, 0 = Không)', [0, 1]) #
if currentSmoker == 1:
    cigsPerDay = st.number_input('Số điếu thuốc mỗi ngày', min_value=0, max_value=100, value=10)
else:
    cigsPerDay = 0
BPmeds = st.selectbox('Sử dụng thuốc hạ huyết áp (1 = Có, 0 = Không)', [0, 1])
prevalentStroke = st.selectbox('Đã từng bị đột quỵ (1 = Có, 0 = Không)', [0, 1])
prevalentHyp = st.selectbox('Huyết áp cao (1 = Có, 0 = Không)', [0, 1])
diabetes = st.selectbox('Bị tiểu đường (1 = Có, 0 = Không)', [0, 1])
totChol = st.number_input('Cholesterol tổng (mg/dL)', min_value=90, max_value=1000, value=200)
sysBP = st.number_input('Huyết áp tâm thu (mmHg)', min_value=70, max_value=350, value=120)
diaBP = st.number_input('Huyết áp tâm trương (mmHg)', min_value=30, max_value=150, value=80)
BMI = st.number_input('Chỉ số BMI', min_value=10.0, max_value=60.0, value=25.0)
heartRate = st.number_input('Nhịp tim (lần/phút)', min_value=40, max_value=150, value=70)
glucose = st.number_input('Mức đường huyết (mg/dL)', min_value=50, max_value=450, value=100)

# Áp dụng log transform
cigsPerDay_log = np.log1p(cigsPerDay)
totChol_log = np.log1p(totChol)
sysBP_log = np.log1p(sysBP)
diaBP_log = np.log1p(diaBP)
BMI_log = np.log1p(BMI)
heartRate_log = np.log1p(heartRate)
glucose_log = np.log1p(glucose)
age_log = np.log1p(age)


# Chuyển dữ liệu thành array
# input_data = np.array([[sex, age, currentSmoker, cigsPerDay, BPmeds, prevalentStroke, 
#                         prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

input_data = pd.DataFrame({
    'male': [sex],  # Đã đổi 'sex' thành 'male'
    'currentSmoker': [currentSmoker],
    'BPMeds': [BPmeds],  # Đã đổi 'BPmeds' thành 'BPMeds'
    'prevalentStroke': [prevalentStroke],
    'prevalentHyp': [prevalentHyp],
    'diabetes': [diabetes],
    'log_cigsPerDay': [cigsPerDay_log],
    'log_totChol': [totChol_log],
    'log_sysBP': [sysBP_log],
    'log_diaBP': [diaBP_log],
    'log_BMI': [BMI_log],
    'log_heartRate': [heartRate_log],
    'log_glucose': [glucose_log],
    'log_age': [age_log]
})

input_data_scaled = scaler.transform(input_data)

# Dự đoán
if st.button('Dự đoán'):
    prediction = model.predict(input_data_scaled)

    prediction_proba = model.predict_proba(input_data_scaled)
    
    # st.write(f'Xác suất không mắc bệnh tim: {(prediction_proba[0][0]*100):.2f}%')
    st.write(f'Xác xuất mắc bệnh tim: {(prediction_proba[0]*100):.2f}%')

    st.subheader('Kết luận:')
    if prediction[0] == 1:
        st.write('Nguy cơ cao mắc bệnh tim.')
    else:
        st.write('Nguy cơ thấp mắc bệnh tim.')

    
    
