import streamlit as st
from db_connection import get_match_by_index

# --- Initialize session state ---
if 'match_index' not in st.session_state:
    st.session_state.match_index = 0
if 'has_guessed' not in st.session_state:
    st.session_state.has_guessed = False

# --- Fetch the current match ---
match = get_match_by_index(st.session_state.match_index)

if match is None:
    st.title("No more matches available!")
    st.stop()

# --- Show the teams ---
st.title('Guess the Winning Team!')

team1 = [match.champ1, match.champ2, match.champ3, match.champ4, match.champ5]
team2 = [match.champ6, match.champ7, match.champ8, match.champ9, match.champ10]

st.subheader('Team 1')
st.write(', '.join(team1))

st.subheader('Team 2')
st.write(', '.join(team2))

# --- Guessing Buttons ---
st.write('Who do you think won?')

col1, col2 = st.columns(2)

with col1:
    if st.button('Team 1 Wins'):
        st.session_state.has_guessed = True
        if match.team1Win:
            st.success('Correct! Team 1 won.')
        else:
            st.error('Wrong! Team 2 actually won.')

with col2:
    if st.button('Team 2 Wins'):
        st.session_state.has_guessed = True
        if not match.team1Win:
            st.success('Correct! Team 2 won.')
        else:
            st.error('Wrong! Team 1 actually won.')

# --- Next Match Button ---
if st.session_state.has_guessed:
    if st.button('Next Match'):
        st.session_state.match_index += 1
        st.session_state.has_guessed = False
        st.rerun()  # Refresh page to load new match
