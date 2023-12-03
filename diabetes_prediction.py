import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/home/devsurya/Machine_Learning_host/diabetes_model.sav', 'rb'))


def diabetes_prediction(input_data):

    input_data_as_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if prediction == 1:
        return('The persion is diabetic')
    else :
        return('persion is not diabetic')

def main():

    st.title("Diabetes Prediciton")

    preg = st.text_input('Number of pregnacies')
    glu = st.text_input('glucose level')
    bp = st.text_input('bloodpressure value')
    sk = st.text_input('skinthickness value')
    insulin = st.text_input('insulin value')
    bmi = st.text_input('Number of bmi')
    dpf = st.text_input('Diabetes pedigree funciton value')
    age = st.text_input('Number of age')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([preg, glu, bp, sk, insulin, bmi, dpf, age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()









