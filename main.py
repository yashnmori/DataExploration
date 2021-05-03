import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

# Web App Title
st.markdown('''
# **The EDA App**

This is the **EDA App** created in Streamlit using the **pandas-profiling** library.

**Credit:** App built in `Python` + `Streamlit` by [Yash Mori](https://instagram.com/y_a_s_h)

---
''')

# Upload CSV data
#with st.sidebar.header('Select your dataset',("wine","iris","breast cancer")):
dataset_name  = st.sidebar.selectbox("select dataset",("Select here","wine","iris","breast cancer"))    

#def get_dataset(dataset_name):
if dataset_name == "Select here":
    st.header("Please Select Dataset")
elif dataset_name == "iris":
    data= load_iris()
elif dataset_name == "wine":
    data= load_wine()
else:
    data = load_breast_cancer()
#X=data.data
#y=data.target
 #   return X,y
df=pd.DataFrame(data=data.data,columns=data.feature_names)
df.head()

    

st.write(df.shape)
pr = ProfileReport(df, explorative=True)
st.header('**Input DataFrame**')
st.write(df)
st.write('---')
st.header('**Pandas Profiling Report**')
st_profile_report(pr)

# Pandas Profiling Report
