import streamlit as st
import predictor

def main():
    st.title("survival of patients with sepsis predictor")

    # Collect user input for age
    age = st.number_input("Enter Age", min_value=1, max_value=150, value=25)

    # Collect user input for gender
    gender_options = ['Male', 'Female']
    gender = st.selectbox("Select Gender", gender_options)

    # Collect user input for episode number
    episode_number = st.number_input("Enter Episode Number", min_value=0, value=1)

    # Submit button to process the data
    if st.button("Submit"):
        process_data(age, gender, episode_number)

def process_data(age, gender, episode_number):
    if gender == "Male":
        gender = 0
    else:
        gender = 1

    result = predictor.predict_proba({"age_years": [age], "sex_0male_1female": [gender], "episode_number": [episode_number]})
    if result[0][0] > 0.5:
        st.error(f"Output: Dead with {round(result[0][0] * 100, 3)}% confidance")
    elif result[0][1] > 0.5:
        st.success(f"Output: Alive with {round(result[0][1] * 100, 3)}% confidance")
if __name__ == "__main__":
    main()