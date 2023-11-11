import streamlit as st
import os
import pickle
import pandas as pd
import shap 

#load model
model_path  = "Artifacts/Catboost_best.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

preprocessor_path = "Artifacts/prop.pkl"
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

st.title("CLIMDES Internship Success Prediction")

#define information files for users to make predictions
age = st.number_input("Enter applicant's age",min_value=18,max_value=40,value=20)
being_student = st.selectbox("Is the applicant as student?",[True,False])

if being_student == True:
    not_student = False
else:
    not_student = True

Internship_Experience = st.number_input("Select internship experience of applicant",min_value=0,max_value=5)
Year_of_Study = st.number_input("What is appliant year of study..select put -1 if applicant is not a student",max_value=5)
Work_Experience = st.number_input("How many years of work experience?",max_value=10,min_value=0)
Research_Experience = st.number_input("How many years of research experience?",max_value=5,min_value=0)
Hours_Dedicated = st.number_input("How many hours of dedication can applicant give",min_value=3,max_value=35)


gender = st.selectbox("Select gender",tuple(["Male", "Female","Non-Binary"]))
Marital_Status = st.selectbox("Marital Status",tuple(["Divorced","Single","Widowed","Married"]))
Internet_Availability = st.selectbox("Internet Availability",tuple(['Dial-Up' ,'High-Speed']))
Availability = st.selectbox("Availability of applicant",tuple(['Full-time' ,'Flexible' ,'Part-time']))
Language_Proficiency = st.selectbox("Language proficiency of applicant",tuple(['Beginner', 'Intermediate' ,'Advanced' ,'Native']))
Continent = st.selectbox("Continent of applicant",tuple(['Australia' ,'South America' ,'North America' ,'Europe', 'Asia' ,'Africa',
 'Antarctica']))
Soft_Skills = st.selectbox("Rate skills of applicant",tuple(['Poor' ,'Good' ,'Excellent' ,'Average']))


features = {'Age':[age],
 'Student':[being_student],
 'Non-student':[not_student],
 'Internship_Experience':[Internship_Experience],
 'Year_of_Study':[Year_of_Study],
 'Work_Experience':[Work_Experience],
 'Research_Experience':[Research_Experience],
 'Hours_Dedicated':[Hours_Dedicated],
 'Gender':[gender],
 'Marital_Status':[Marital_Status],
 'Internet_Availability':[Internet_Availability],
 'Availability':[Availability],
 'Language_Proficiency':[Language_Proficiency],
 'Continent':[Continent],
 'Soft_Skills':[Soft_Skills],
 }
data = pd.DataFrame(features)

transformed_data = preprocessor.transform(data)

if st.button("Predict"):
    prediction = model.predict(transformed_data)[0]
    conf = model.predict_proba(transformed_data)[0][0]
    if prediction == 1:
        st.write("Applicant is likely to complete the internship successfully")
    
    else:

        st.write("Applicant has a low chance of succeding in the internship")


    st.write(f"Confidence: **{conf:.2f}%**")

        # Use SHAP to explain the model's prediction for the single row
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_data)

    # Summary plot for the single row
    shap.summary_plot(shap_values, transformed_data, show=False)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
