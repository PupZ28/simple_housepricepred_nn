import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# โหลดหรือเทรน pipeline ของโมเดล
@st.cache_resource
def load_pipeline():
    try:
        model = joblib.load("model_pipeline.pkl")
    except:
        # การโหลดข้อมูลและการพรีโปรเซสซิ่ง
        df = pd.read_csv("data.csv")  # เปลี่ยนเป็น path ของข้อมูลการเทรนที่ใช้
        df = df.drop(columns=['Address', 'Date'])
        df = pd.get_dummies(df, columns=['Method', 'SellerG', 'CouncilArea', 'Regionname', 'Season', 'Suburb'], drop_first=True)
        y = df['Price']
        X = df.drop('Price', axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # สร้างโมเดล
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

        # บันทึก pipeline
        joblib.dump({"model": model, "scaler": scaler}, "model_pipeline.pkl")
        
    return joblib.load("model_pipeline.pkl")

pipeline = load_pipeline()
model, scaler = pipeline["model"], pipeline["scaler"]

# Title and File Upload
st.title("House Price Prediction App")

# ฟอร์มสำหรับการพยากรณ์
st.subheader("กรอกข้อมูลเกี่ยวกับอสังหาริมทรัพย์เพื่อการพยากรณ์:")
feature_inputs = {}
# แทนที่ด้วยชื่อและชนิดของฟีเจอร์จริง ๆ
feature_inputs['Rooms'] = st.number_input("จำนวนห้อง", min_value=1, max_value=10, value=3)
feature_inputs['Distance'] = st.number_input("ระยะห่างจาก CBD (กม.)", min_value=0.0, value=5.0)
feature_inputs['Landsize'] = st.number_input("ขนาดที่ดิน (ตร.ม.)", min_value=0.0, value=500.0)
feature_inputs['BuildingArea'] = st.number_input("ขนาดพื้นที่สิ่งปลูกสร้าง (ตร.ม.)", min_value=0.0, value=150.0)

# เตรียมข้อมูลอินพุตและพยากรณ์
if st.button("ทำนายราคาบ้าน"):
    input_df = pd.DataFrame([feature_inputs])
    input_scaled = scaler.transform(input_df)
    price_pred = model.predict(input_scaled)[0][0]
    
    st.write(f"ราคาที่คาดการณ์: ${price_pred:,.2f}")

# การแสดงผลตัวอย่างการพยากรณ์จากชุดทดสอบ
uploaded_file = st.file_uploader("หรืออัปโหลดไฟล์ CSV สำหรับการพยากรณ์แบบกลุ่ม", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # ทำการพรีโปรเซสซิ่งและพยากรณ์ตามที่โค้ดเดิมของคุณกำหนดไว้
