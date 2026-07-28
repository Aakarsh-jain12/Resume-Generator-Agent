import streamlit as st 
import os
import time
import langchain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
import pytesseract as pyt
import numpy as np
from langchain.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from PIL import Image 

#=========================FRONTEND========================
st.title("AI RESUME maker & JOB APPLY AGENT")
st.image("https://framerusercontent.com/images/tzBelfM3npZvuhrC3Rt8As8lVzo.png?width=1536&height=1024")

GOOGLE_API_KEY = st.sidebar.text_input("Google Api key", type = 'password')
GROQ_API_KEY = st.sidebar.text_input("GROQ Api key", type = 'password')
TAVILY_API_KEY = st.sidebar.text_input("TAVILY Api key", type = 'password')

if not (GOOGLE_API_KEY) and not (GROQ_API_KEY) and not (TAVILY_API_KEY):
  st.sidebar.warning("Pass api keys")
  st.stop
else:
  st.success("API KEYS LAODED")  


#========================MODEL AND AGENT CODE======================
# tool 1
def search_latest_news_job(query):
  """This function helps to get
  latest news or latest jobs
  related to user given query
  using tavily"""

  from tavily import TavilyClient
  client = TavilyClient(api_key = TAVILY_API_KEY)
  return client.search(query)



# step 4: Model and agent creation
model1 = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    google_api_key = GOOGLE_API_KEY
)

model2 = ChatGroq(
    model = "qwen/qwen3.6-27b",
    api_key = GROQ_API_KEY
)


#===================Agent WIth tool==========================
agent = create_agent(
    model = model1,       # can be model2 also,
    tools = [search_latest_news_job]
)



def prompt_generator():
  prompt = """You are helpful AI Resume
  maker, I want you to use chain-of-thoughts
  and give detailed prompt for model
  where user want to generate resume
  for fresher or experienced one
  in HTML format, you have to give proper
  set of instructions, and make sure to keep
  design professional"""

  response = model1.invoke(prompt)
  prompt_ans = response.content[-1]['text']
  # print(prompt_ans)

  file_name = 'prompt.txt'
  with open(file_name, 'w') as f:
    f.write(prompt_ans)

prompt_generator()



# Final_Agent

# Tool 2

def prompt_reader():
  with open('prompt.txt', 'r') as f:
    prompt = f.read()
  return prompt
prompt = """I want complete Professional
Resume With dynamic design using Advanced CSS and JS
and must show user input details
system instructions: Only give HTML code as output """

final_prompt = prompt + prompt_reader()

#==============================IAMGE UPLOADAR================================

FILE = st.sidebar.file_uploader(
  "choose an image file",
  type=["jpg","jpeg","png","webp"]
)

if FILE is not None:
  try:
    image = Image.open(FILE)

    st.sidebar.image(image,
                     caption="Uploaded Image",
                     use_container_width=True)

    if image.mode in ("RGBA", "P"):
      image = image.convert("RGB")

    base_name = os.path.splitext(FILE.name)[0]
    save_path = f"{base_name}.jpg"
    
    image.save(save_path, "JPEG")
    st.sidebar.success(f" Image successfully saved as `{save_path}`!")

  except Exception as e:
    st.error(f"Error processing image: {e}")

#change this when required new resume by user, pass details

user_info = st.text_input("Give Your Information")
user_photo = st.sidebar.file_uploader("upload pic", type = 'image/jpeg')


user_query = f"""Give resume for python Developer.
   user details: {user_info} 
    use user profile image from given url: {user_photo}"""

final_query = final_prompt + user_query

if st.button("Generate Resume"):
  with st.spinner("Agent creating Resume... "):
    response = agent.invoke({'messages':[{'role':'user',"content":final_query}]})
    code = response['messages'][-1].content[-1]['text']

    st.html(code, width="stretch", unsafe_allow_javascript=True)


