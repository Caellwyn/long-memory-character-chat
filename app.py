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


def format_history(history):
    """Format the conversation history for display.  Returns a string."""
    report = ''
    for message in reversed(history[:]):
        if message['role'] == 'user':
            report += (f"\n\nYOU: {message['content'].replace(agent.prefix, '')}")
        else:
            report += (f"\n\nCHARACTER: {message['content']}")
    return report
 
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

if 'pickled_agent' not in st.session_state:
    st.session_state['pickled_agent'] = None


# Set the title
st.title('Chat with a Character!')

# set the temperature for the model
temperature = st.sidebar.slider('Creativity', min_value=0.0, max_value=1.0, step=0.1, value=0.1)

# add a button to clear the conversation history
st.sidebar.button('New conversation', on_click=clear_history,
                   use_container_width=False)

# add a button to save the character and conversation
st.sidebar.button('Save Conversation', on_click=save_character)

# add a button to download the character and conversation
if st.session_state['pickled_agent']:
    st.sidebar.download_button(
        label='Download Character and Conversation',
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
                        max_chars=500, help='Describe the character', key='character')
st.button('Set Character', on_click=set_character, args=[character])

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
