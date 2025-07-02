import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/logistic_final_titanic.pkl")

st.title("🛳️ Dự đoán hành khách sống sót Titanic")
st.write("Nhập thông tin hành khách để dự đoán khả năng sống sót.")

age = st.number_input("Tuổi", min_value=0.0, max_value=100.0, value=30.0)
fare = st.number_input("Giá vé", min_value=0.0, value=50.0)
sex = st.selectbox("Giới tính", ["male", "female"])
pclass = st.selectbox("Hạng ghế (Pclass)", [1, 2, 3])
embarked = st.selectbox("Cảng lên tàu (Embarked)", ["S", "C", "Q"])
title = st.selectbox("Danh xưng (Title)", ["Mr", "Mrs", "Miss", "Master", "Other"])
family_cat = st.selectbox("Tình trạng gia đình", ["Alone", "Small", "Large"])


if st.button("🎯 Dự đoán"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Fare": fare,
        "Sex": sex,
        "Pclass": pclass,
        "Embarked": embarked,
        "Title": title,
        "Family_Cat": family_cat
    }])

    # Dự đoán
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Kết quả
    if prediction == 1:
        st.success(f"✅ Sống sót với xác suất {prob:.2%}")
    else:
        st.error(f"❌ Không sống sót với xác suất {(1 - prob):.2%}")



st.markdown("---")
st.markdown(
    "<h4 style='text-align: center;'>👨‍💻 Người thực hiện: <b>Võ Nguyễn Nhật Duy</b></h4>",
    unsafe_allow_html=True
)