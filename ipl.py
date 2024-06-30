import streamlit as st
import pickle
import pandas as pd

st.title("IPL Winner predictor")

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

city=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']
data=pickle.load(open('data_ipl1.pkl','rb'))
pipe=pickle.load(open('pipe_ipl1.pkl','rb'))



batting=st.selectbox('batting team',sorted(teams))

bowling=st.selectbox('bowling team',sorted(teams))

selected_city=st.selectbox("City",sorted(city))

target = st.number_input('Target')

score = st.number_input('Score')

overs = st.number_input('Overs completed')

wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):

    runs_left=target-score
    balls_left=120-(overs*6)
    wickets_left=10-wickets
    crr=score/overs
    rr=(runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting], 'bowling_team': [bowling], 'city': [selected_city],
                             'total_runs_x': [target], 'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets_left], 'crr': [crr], 'rr': [rr]})
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header(batting + "- " + str(round(win * 100)) + "%")
    st.header(bowling + "- " + str(round(loss * 100)) + "%")


