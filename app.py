import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers
from aiagent import AIAgent


# get the websocket headers and session id
try:
    headers = _get_websocket_headers()
    session_id = headers.get("Sec-Websocket-Key")
except:
    session_id = 'default'

@st.cache_resource
def get_agent(session_id, model='open-mistral-7b', ):
    """Create an AI agent.  Returns an AIAgent object."""
    agent = AIAgent(model=model)
    print('creating the ai agent')
    return agent


def query_agent(prompt, temperature=0.1, top_p=0.0):
    """Query the AI agent.  Returns a string."""
    try:      
        st.session_state['agent'].query(prompt, 
                    temperature=temperature,
                    top_p=top_p)
    except Exception as e:
        print('failed to query the agent')
        print(e)

def clear_history():
    """Clear the AI's memory.  Returns nothing."""
    st.session_state['agent'].clear_history()

def set_character(character):
    """Set the AI's character.  Returns nothing."""
    st.session_state['agent'].set_character(character)

def save_character():
    """Save the AI's character and conversation history.  Returns nothing."""
    st.session_state['pickled_agent'] = st.session_state['agent'].save_agent()

def load_character(file):
    """Load the AI's character and conversation history.  Returns nothing."""
    st.session_state['agent'].load_agent(file)
    st.session_state['character_description'] = st.session_state['agent'].character

def change_model():
    """Change the AI's model.  Returns nothing."""
    if 'agent' in st.session_state:
        st.session_state['agent'].set_model(st.session_state['model_name'])
        st.session_state['agent'].set_summary_model(st.session_state['model_name'])
    else:
        st.session_state['agent'] = get_agent(session_id)


# Set the title
st.title('Chat with a Character!')

# Disclaimer
st.write('''This app allows you to chat with a character.  You can set the character description, save and load conversations, and clear the conversation history.
         This character will have a very long memory, and should remember the conversation even after many messages.  It's not perfect, but better than most!
         The character the language model selected below to generate responses.
         \nAll responses are meant for entertainment only.  The character is not a real person and does not have real emotions or thoughts.
         The character is not a substitute for professional advice.  Please do not share personal information with the character.
         The character and developer are not responsible for any actions you take based on its responses.
         Responses should not be considered factual in any way.
         Please use this app responsibly.  Enjoy!''')


with st.container(border=True):
    # set the temperature for the model
    st.markdown('#### Model Settings')
    temperature = st.slider('Creativity', min_value=0.1, max_value=1.0, step=0.1, value=0.1)
    top_p = st.slider('Freedom', min_value=0.0, step=.05, max_value=1.0, value=0.0)
    # set the top_p value
    top_p = 1 - top_p
    if top_p == 1 or 0:
        top_p = None

    # set the model
    st.radio("Model to use (in Ascending Order of Cost)", horizontal=True,
            options=['open-mistral-7b', 'gpt-3.5-turbo-0125', 'open-mixtral-8x7b'],
            index=0,
            key='model_name', on_change=change_model)

# get the agent
if 'agent' not in st.session_state:
    st.session_state['agent'] = get_agent(session_id, model=st.session_state['model_name'])
else:
    change_model()

# if there is no pickled agent in the session state, set it to None
if 'pickled_agent' not in st.session_state:
    st.session_state['pickled_agent'] = None

with st.container(border=True):
    st.markdown('#### Character Settings')
    # set the character with a text input and button
    character = st.text_area('The character is...', value=st.session_state['agent'].character,
                            max_chars=500, help='Describe the character', key='character', height=100)
    set_character(character)
    # add a button to clear the conversation history
    st.button('Reset conversation', on_click=clear_history,
                       use_container_width=False)

st.markdown('#### Chat with the Character')

# Create chat input
with st.expander("Input Messages",expanded=True):
    if prompt := st.chat_input("Your message here", max_chars=500):
        with st.spinner("Thinking..."):
            query_agent(prompt, 
                        temperature=temperature,
                        top_p=top_p)
            
# display the conversation history
with st.container(height=500):
    for message in reversed(st.session_state["agent"].chat_history):
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'].replace(st.session_state['agent'].prefix, ''))
            else:
                st.markdown(message['content'])


st.sidebar.markdown('#### Save, Load, and Upload Coversations')
# add a button to save the character and conversation
st.sidebar.button('Save Conversation', on_click=save_character)    

# if there is a saved conversation, add buttons to reload and download the character and conversation
if st.session_state['pickled_agent']:
    # add a button to reload the character and conversation
    st.sidebar.button('Reload Conversation', on_click=load_character, args=[st.session_state['pickled_agent']])

    # add a button to download the character and conversation
    st.sidebar.download_button(
        label='Download Conversation',
        data=st.session_state['pickled_agent'],
        file_name="saved_character.pkl",
        mime="application/octet-stream")

# add a button to upload a character and conversation
with st.sidebar.form('upload_character', clear_on_submit=True):
    uploaded_file = st.file_uploader(":floppy_disk: **Upload a saved conversation**", 
                                     type=['pkl'], accept_multiple_files=False,)
    
    submit_button = st.form_submit_button('Upload')
    if submit_button:
        if uploaded_file is not None:
            pkl = uploaded_file.getvalue()
            load_character(pkl)
            st.rerun()
# write descriptive statistics on the sidebar
st.sidebar.write(f'Total current memory tokens: {st.session_state["agent"].current_memory_tokens}')
st.sidebar.write(f'Total cost of this conversation is: {st.session_state["agent"].total_cost}')
st.sidebar.write(f'Total tokens sent is: {st.session_state["agent"].total_tokens}')
st.sidebar.write(f'Average number of tokens per message is: {st.session_state["agent"].average_tokens}')
st.sidebar.write(f'Average cost per message is: {st.session_state["agent"].average_cost}')
st.sidebar.write(f'Total number of messages to and from the character: {len(st.session_state["agent"].chat_history)}')