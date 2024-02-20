import streamlit as st
from aiagent import AIAgent

from streamlit.web.server.websocket_headers import _get_websocket_headers
import streamlit as st

headers = _get_websocket_headers()

session_id = headers.get("Sec-Websocket-Key")

@st.cache_resource
def get_agent(session_id):
    agent = AIAgent(chat_model='gpt-3.5-turbo')
    print('creating the ai agent')
    return agent

def format_history(history):
    report = ''
    for message in history[1:]:
        if message['role'] == 'user':
            report += (f"\n\nYOU: {message['content'].replace(agent.prefix, '')}")
        else:
            report += (f"\n\nCHARACTER: {message['content']}")
    return report
 
def query_agent(prompt, temperature=0.1):
    try:      
        agent.query(prompt, 
                    temperature=temperature
                    )

    except Exception as e:
        print(e)
        return "Something went wrong..."

def clear_history():
    agent.clear_history()

def set_character(character):
    agent.set_character(character)


st.title('Chat with a Character!')

temperature = st.sidebar.slider('Creativity', min_value=0.0, max_value=1.0, step=0.1, value=0.1)

st.sidebar.button('New conversation', on_click=clear_history,
                   use_container_width=False)

agent = get_agent(session_id)

character = st.text_input('Character Description', value='A friendly old man')

character_set = st.button('Change Character')

if character_set and character:
    set_character(character)

if character:
    agent.set_character(character)

prompt = st.text_input(label="Your message here",
                       max_chars=100,
                       value='',
                       help="Write your message here",
                       key='user_query',
                       placeholder="Write your message here")

queried = st.button('Submit your message')

if queried and prompt:
    current_response = st.markdown("Thinking...")
    query_agent(prompt, 
                temperature=temperature
                )
    current_response.write(agent.response)

st.sidebar.markdown(f'The conversation so far: {format_history(agent.chat_history)}')





