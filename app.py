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
    for message in history[1:]:
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

# Set the title
st.title('Chat with a Character!')

# set the temperature for the model
temperature = st.sidebar.slider('Creativity', min_value=0.0, max_value=1.0, step=0.1, value=0.1)

# add a button to clear the conversation history
st.sidebar.button('New conversation', on_click=clear_history,
                   use_container_width=False)

# get the agent
agent = get_agent(session_id)

# set the character with a text input and button
character = st.text_input('Character Description', value='A friendly old man')
character_set = st.button('Change Character')
if character_set and character:
    set_character(character)

# add a text input and button for the user's message
prompt = st.text_input(label="Your message here",
                       max_chars=100,
                       value='',
                       help="Write your message here",
                       key='user_query',
                       placeholder="Write your message here")
queried = st.button('Submit your message')

# if the user has created a prompt and pressed the query button, then query the AI
if queried and prompt:
    current_response = st.markdown("Thinking...")
    query_agent(prompt, 
                temperature=temperature
                )
    current_response.write(agent.response)

# display the conversation history on the sidebar
st.sidebar.markdown(f'The conversation so far: {format_history(agent.chat_history)}')





