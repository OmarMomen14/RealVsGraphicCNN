import streamlit as st
import numpy as np
import pandas as pd
import SessionState
import ImageClassifier as ic
from PIL import Image
import requests

# """
# # Get Your Text here @ 

# This model automatically retrieves closest matching headlines and taglines. 
# Headlines and taglines come from:

# - curated list or
# - auto-generated if non found in the list.
# """



st.set_option('deprecation.showfileUploaderEncoding', False)
text_one = st.sidebar.text_input(label="Enter keyword(s)!", key="input_1")
text_two = st.sidebar.text_input(label="Enter a headline!", key="input_2")


session_state = SessionState.get(name="", button_sent=False)
button_sent = st.sidebar.button('Get Images', key='input_one')

# first time is button interaction, next time use state to go to multiselect
if button_sent or session_state.button_sent:
    headline = []
    session_state.button_sent = True
    
    """
    ## Your inputs:
    """
    st.write("**Keywords**: ",text_one)
    st.write("**Headline**: ",text_two)


    response = requests.post('http://3.19.120.21:8881/get_email_image', json={"input_four": text_one,
                                                                              "input_five": text_two}                                                
                            )

    if response.status_code == 200:
        imagesUrls = response.json()['asset_links']
    
        col1, col2 = st.beta_columns(2)
        imagesList = []
        notRealImg = Image.open('notReal.jpg')

        col1.header("Without CNN reality check")
        for url in imagesUrls:
            image = Image.open(requests.get(url, stream=True).raw)
            imagesList.append(image)
            col1.image(image, use_column_width=True)

        
        col2.header("After CNN reality check")
        results = ic.are_real_URLS(imagesUrls)
        for (result, image) in zip(results, imagesList):
            if result:
                col2.image(image, use_column_width=True)
            else:
                col2.image(notRealImg, use_column_width=True)


    else:
        st.write("Retrieving images from the API failed")
    
else:
    st.write("Please key in your input.")

