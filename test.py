import streamlit as st
import pandas as pd
import numpy as np
import pickle
pipe=pickle.load(open('pipe.pkl','rb'))
st.title("IPL Win Predictor")
teams=['Royal Challengers Bangalore',
 'Kings XI Punjab',
 'Mumbai Indians',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Pune Warriors',
 'Sunrisers Hyderabad',
 'Delhi Capitals']
col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox("Select the bowling team",sorted(teams))
target=st.number_input('target')
col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('score')
with col4:
    overs=st.number_input('Overs completed')
with col5:
    wickets=st.number_input('Wickets out')
 
balls_left=120-(overs*6)
r = (target)-(score)    
wickets_left=10-wickets
if overs != 0:
     crr = score / overs
else:
    crr = 0  
rrr=(r*6)/balls_left 
     
     
      # Or another fallback value
   
input_df = pd.DataFrame({
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'runs_left': [r],
    'balls_left': [balls_left],
    'wickets': [wickets_left],
    'total_runs_x': [target],  # Ensure this matches the expected name
    'crr': [crr],
    'rrr': [rrr]
})
st.table(input_df)
result=pipe.predict_proba(input_df)
st.text(result)
loss=result[0][0]
win=result[0][1]
st.text(batting_team+'-'+str(round(win*100))+"%")
st.text(bowling_team+'-'+str(round(loss*100))+"%")
