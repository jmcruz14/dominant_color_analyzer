import streamlit as st
from index import process_vid

# Functions
def start_process_vid(video_file):
  try:
    if video_file is None:
      raise Exception
    else:
      print('starting-process-vid')
      process_vid(video_file)
  except:
    st.toast("Upload a video first.")

st.title("Dominant Color Analyzer")
st.subheader("This app serves to scan video files and return the five most dominant colors in said video file.")
st.divider()

video_file = st.file_uploader("Upload video here")
st.button("Process video", on_click=start_process_vid(video_file), type="primary")
# insert input here

# insert output here

st.divider()
st.caption('Copyright - Jay Cruz, 2024')