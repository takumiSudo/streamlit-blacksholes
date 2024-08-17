#!/usr/bin/env python3

import streamlit as st
import pandas as pd

st.title("My first streamlit app")
st.write("Welcome to my StreamLit APP")

user_input = st.text_input('Enter Custom Message', 'Hello StreamLit')
st.write('Customized Message:', user_input)

st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [12, 113, 41, 42]
}))
