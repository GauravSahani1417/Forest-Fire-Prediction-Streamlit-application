# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 09:57:07 2020

@author: gaurav sahani
"""
import streamlit as st
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    print(type(pred))
    return float(pred)

def main():
    st.title("Forest Fire Prediction")
    html_temp="""
    <h3><font color="black">Made By: Gaurav R. Sahani</font></h3>
    <div style="background-color:#000706 ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction Application</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)    
    
    oxygen=st.slider("Select Oxygen Content", 0, 100)
    humidity=st.slider("Select Humidity Content ", 0, 100)
    temperature=st.slider("Select Temperature Value", 0, 100)
    
    safe_html="""
      <div style="background-color:#BEF43F;padding:10px >
        <h2 style="color:white;text-align:center;">The Forest is Safe from Fire</h2>
        </div>
    """
    
    danger_html="""
      <div style="background-color:#F08080;padding:10px > 
        <h2 style="color:Red;text-align:center;">The Forest is in Danger!</h2>
        </div>
    """

    if st.button("Predict"):
        output=predict_forest(oxygen,humidity,temperature)
        st.success('The probability of fire taking place is {}'.format(output))
        
        if output>0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
            
if __name__=='__main__':
    main()            
        
        
        
        
        
        
        
        
        