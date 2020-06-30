import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image  

html_temp = """
    <div style="background-color:#010200;padding:10px">
    <h1 style="color:white;text-align:center;">Forest Fire Prediction</h1>
    </div>
    """    
st.markdown(html_temp,unsafe_allow_html=True)

image = Image.open('forest.png') 

st.image(image, use_column_width=True,format='PNG')

def user_input_features():
    Oxygen = st.slider('Oxygen', 0, 90)
    Temperature = st.slider('Temperature', 0, 60)
    Humidity = st.slider('Humidity', 0, 100)
    data = {'Oxygen': Oxygen,
            'Temperature': Temperature,
            'Humidity': Humidity}
    features = pd.DataFrame(data, index=[0])
    return features

st.subheader('Please Enter Input Through Slider')
df = user_input_features()     
        
st.subheader('User Input parameters')
st.write(df)

df1=pd.read_csv('forest.csv')
df1.drop(['Area'],axis=1,inplace=True)

X=df1[['Oxygen','Temperature','Humidity']]
y=df1[['Target']]

df1.head()

clf = RandomForestClassifier()
clf.fit(X, y.values.ravel())

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction Probability')
st.write('**Forest is under Danger**:',prediction_proba[0][0])
st.write('**Forest is Not under Danger**:',prediction_proba[0][1])

html_temp1 = """
    <div style="background-color:#010200">
    <p style="color:white;text-align:center;" >Designe & Developed By: <b>Gaurav R. Sahani</b> </p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)



        
        
        
        
        
        
        
        