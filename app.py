
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Employee Burnout Prediction", layout="centered")

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Employee Burnout Prediction System")
st.write("Enter employee details below to estimate burnout risk.")

work_hours = st.number_input("Work Hours Per Day", 0.0, 24.0, 8.0)
screen_time = st.number_input("Screen Time (Hours)", 0.0, 24.0, 6.0)
meetings = st.number_input("Meetings Per Day", 0, 20, 2)
breaks = st.number_input("Breaks Per Day", 0, 20, 3)
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
tasks_completed = st.number_input("Tasks Completed", 0, 100, 10)
day_type = st.selectbox("Day Type", ["Weekday","Weekend"])
day_type_val = 0 if day_type=="Weekday" else 1

if st.button("Predict Burnout Risk"):
    data = np.array([[work_hours,screen_time,meetings,breaks,sleep_hours,tasks_completed,day_type_val]], dtype=float)
    scaled = scaler.transform(data)
    pred = int(model.predict(scaled)[0])
    labels = {0:"Low Burnout Risk",1:"Medium Burnout Risk",2:"High Burnout Risk"}
    st.success("Prediction: " + labels.get(pred,"Unknown"))
