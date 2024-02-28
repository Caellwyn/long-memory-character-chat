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
def get_agent(session_id):
    """Create an AI agent.  Returns an AIAgent object."""
    agent = AIAgent(chat_model='gpt-3.5-turbo')
    print('creating the ai agent')
    return agent


def query_agent(prompt, temperature=0.1):
    """Query the AI agent.  Returns a string."""
    try:      
        agent.query(prompt, 
                    temperature=temperature
                    )

    except Exception as e:
        print(e)
        return "Something went wrong..."

def clear_history():
    """Clear the AI's memory.  Returns nothing."""
    agent.clear_history()

def set_character(character):
    """Set the AI's character.  Returns nothing."""
    agent.set_character(character)

def save_character():
    """Save the AI's character and conversation history.  Returns nothing."""
    st.session_state['pickled_agent'] = agent.save_agent()

def load_character(file):
    """Load the AI's character and conversation history.  Returns nothing."""
    agent.load_agent(file)
    st.session_state['character_description'] = agent.character

# get the agent
agent = get_agent(session_id)

# if there is no pickled agent in the session state, set it to None
if 'pickled_agent' not in st.session_state:
    st.session_state['pickled_agent'] = None

# Set the title
st.title('Chat with a Character!')

# Disclaimer
st.write('''This app allows you to chat with a character.  You can set the character description, save and load conversations, and clear the conversation history.  
         The character uses the GPT-3.5-turbo model to generate responses.
         All responses are meant for entertainment only.  The character is not a real person and does not have real emotions or thoughts.  
         The character is not a substitute for professional advice.  Please do not share personal information with the character.  
         The character and developer are not responsible for any actions you take based on its responses.  
         Responses should not be considered factual in any way. 
         Please use this app responsibly.  Enjoy!''')

# add a button to clear the conversation history
st.button('New conversation', on_click=clear_history,
                   use_container_width=False)

# set the temperature for the model
temperature = st.slider('Creativity', min_value=0.0, max_value=1.0, step=0.1, value=0.1)

# add a sidebar with instructions for saving and loading conversations
st.sidebar.markdown('''### Saving and Loading Conversations.  
                    \n To save a conversation at the current message, press `Save Conversation`.  This saves both the current character and the conversation.
                    \n To load a conversation, press `Load Conversation` 
                    \n You can only have one conversation saved at a time. And it will be lost if this page is refreshed, unless you download the conversation file.
                    \n In order to save conversations between sessions, you can download the conversation with the `Download Conversation` buttion which only appears after saving.
                    \n You can upload previous conversations at any time, even after refreshing the page by dropping the downloaded .pkl file in the `Upload a saved conversation` box and pressing `Upload`.''')

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

# set the character with a text input and button
character = st.text_area('Set Character Description', value=agent.character,
                        max_chars=500, help='Describe the character', key='character', height=100)
st.button('Set Character', on_click=set_character, args=[character])
st.write('The character creator is currently a little wonky.  You may need to press the button a few times to get the character to update.  It also may not appear to update when you upload a previous conversion, but you should have your saved character back, even if it doesn\'t look like it here. Sorry for the inconvenience.  I am working on it!')

# Create chat input
with st.expander("Input Messages",expanded=True):
    if prompt := st.chat_input("Your message here", max_chars=500):
        with st.spinner("Thinking..."):
            query_agent(prompt, 
                        temperature=temperature
                        )
            
# display the conversation history
with st.container(height=500):
    for message in reversed(agent.chat_history):
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'].replace(agent.prefix, ''))
            else:
                st.markdown(message['content'])

# write descriptive statistics on the sidebar
st.sidebar.write(f'Total current memory tokens: {agent.current_memory_tokens}')
st.sidebar.write(f'Total cost of this conversation is: {agent.total_cost}')
st.sidebar.write(f'Total tokens sent is: {agent.total_tokens}')
st.sidebar.write(f'Average number of tokens per message is: {agent.average_tokens}')
st.sidebar.write(f'Average cost per message is: {agent.average_cost}')
st.sidebar.write(f'Total number of messages to and from the character: {len(agent.chat_history)}')