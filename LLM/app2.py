import streamlit as st 
from dotenv import load_dotenv 
load_dotenv()
import os 
import google.generativeai as genai 

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
st.title('Chat with Leyla')
model=genai.GenerativeModel('gemini-1.5-pro-latest')
chat=model.start_chat(history=[])
soru=st.text_input("You:")
if st.button('Sor'):
    response=chat.send_message(soru)
    st.write(response.text)
    st.write(chat.history)