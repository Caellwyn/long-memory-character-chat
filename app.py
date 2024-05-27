import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers
from aiagent import AIAgent
import os

st.set_page_config(layout="wide")

# get the websocket headers and session id
try:
    headers = _get_websocket_headers()
    session_id = headers.get("Sec-Websocket-Key")
except:
    session_id = 'default'

try:
    nsfw_password = st.secrets['CHAT_NSFW_PASSWORD']
except Exception as e:
    nsfw_password = os.getenv('CHAT_NSFW_PASSWORD')

@st.cache_resource
def get_agent(session_id, model='open-mistral-7b', ):
    """Create an AI agent.  Returns an AIAgent object."""
    agent = AIAgent(model=model)
    print('creating the ai agent')
    return agent

def query_agent(prompt, temperature=0.1, top_p=0.0, frequency_penalty=0, presence_penalty=0):
    """Query the AI agent.  Returns a string."""
    try:      
        st.session_state['agent'].query(prompt, 
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty)
    except Exception as e:
        print('failed to query the agent')
        print(e)

def clear_history():
    """Clear the AI's memory.  Returns nothing."""
    st.session_state['agent'].clear_history()

def set_character():
    """Set the AI's character.  Returns nothing."""
    if 'character' in st.session_state:
        st.session_state['agent'].set_character(st.session_state['character']) 

def set_location():
    """Set the AI's location.  Returns nothing."""
    if 'location' in st.session_state:
        st.session_state['agent'].set_location(st.session_state['location'])

def set_user_name():
    """Set the AI's user name.  Returns nothing."""
    if 'user_name' in st.session_state:
        st.session_state['agent'].set_user_name(st.session_state['user_name'])

def set_character_name():
    """Set the AI's character name.  Returns nothing."""
    if 'character_name' in st.session_state:
        st.session_state['agent'].set_character_name(st.session_state['character_name'])

def save_character():
    """Save the AI's character and conversation history.  Returns nothing."""
    st.session_state['pickled_agent'] = st.session_state['agent'].save_agent()

def load_character(file):
    """Load the AI's character and conversation history.  Returns nothing."""
    st.session_state['agent'].load_agent(file)
    # st.session_state['character'] = st.session_state['agent'].character

def change_model():
    """Change the AI's model.  Returns nothing."""
    if 'agent' in st.session_state:
        st.session_state['agent'].set_model(st.session_state['model_name'])
    else:
        if 'model_name' in st.session_state:
            st.session_state['agent'] = get_agent(session_id, model=st.session_state['model_name'])
        else:
            st.session_state['agent'] = get_agent(session_id)

def set_nsfw():
    """Set the AI's NSFW mode.  Returns nothing."""
    if 'nsfw' in st.session_state:
        st.session_state['agent'].nsfw = st.session_state['nsfw']

def format_model_label(model):
    labels = [('gemini-pro (free, but sometimes doesn\'t respond)','gemini-pro'),
            ('openchat 3.5','openchat/openchat-3.5-1210'),
            ('Qwen1.5 7b','Qwen/Qwen1.5-7B-Chat'),
            ('StripedHyena Nous 7b','togethercomputer/StripedHyena-Nous-7B'),
            ('Nous Capybara 7b','NousResearch/Nous-Capybara-7B-V1p9'),
            ('Vicuna 7b v1.5','lmsys/vicuna-7b-v1.5'),
            ('Mistral 7B Instruct v0.1','mistralai/Mistral-7B-Instruct-v0.1'),
            ('Meta Llama 2 7b','meta-llama/Llama-2-7b-chat-hf'),
            ('Nous Hermies Llama 2 13b','NousResearch/Nous-Hermes-Llama2-13b'),
            ('Meta Llama 2 13b','meta-llama/Llama-2-13b-chat-hf'),
            ('WizardLM 13b v1.2','WizardLM/WizardLM-13B-V1.2'),
            ('GPT 3.5 Turbo','gpt-3.5-turbo-0125'),
             ('Meta Llama-3 8b', 'meta-llama/Llama-3-8b-chat-hf')]
    for label in labels:
        if model == label[1]:
            return label[0]
    return model

# Set the title
st.title('Chat with a Character!')


with st.expander('Disclaimer', expanded=False):
# Disclaimer
    st.write('''This app allows you to chat with a character.  You can set the character description, save and load conversations, and clear the conversation history.
            This character will have a very long memory, and should remember details about you conversation even after many messages.  It's not perfect, but better than most!
            You can set the temperature and top_p through 'creativity' and 'freedom' respectively.  You can also choose from one of 3 base models.  These can be changed on the fly.
            \n * All responses are for entertainment only.  

            \n * The character is not a real person and does not have real emotions or thoughts.

            \n * While I do not keep any of your information after you navigate away or refresh the page, it is being sent to 3rd party servers for processing.  Please do not share any personal information with the character.

            \n * The character is not a substitute for professional advice.

            \n * The character and developer are not responsible for any actions you take based on its responses.

            \n * Responses should not be considered factual in any way.

            \n * While I take steps to try to filter model responses, I cannot guarantee that all responses will be appropriate.

            \n * If you are under 18, please navigate away from this page.

            \n * Please use this app responsibly and have lots of fun!  Enjoy!''')

with st.sidebar:

    # Create the model settings

    with st.container(border=True):
        # set the temperature for the model
        st.markdown('#### Model Settings')
        temperature = st.slider('Temperature: gives the model more freedom to be creative', min_value=0.0, max_value=1.0, step=0.05, value=0.0)
        top_p = st.slider('Top P: allows the model to choose from a larger selection of possible responses', min_value=0.0, step=.05, max_value=1.0, value=0.0)
        frequency_penalty = st.slider('Frequency Penalty: helps make responses less repetitive', min_value=-2.0, step=.05, max_value=2.0, value=.0)
        presence_penalty = st.slider('Presence Penalty: Encourages more diverse vocabulary', min_value=-2.0, step=.05, max_value=2.0, value=.0)
        # set the top_p value
        top_p = 1 - top_p
        if top_p == 1 or 0:
            top_p = None

        # set the model
        st.markdown('### Choose a model')
        st.radio("Models", horizontal=False,
                options=['gemini-pro',
                        'openchat/openchat-3.5-1210',
                        'Qwen/Qwen1.5-7B-Chat',
                        'togethercomputer/StripedHyena-Nous-7B',
                        'lmsys/vicuna-7b-v1.5',
                        'mistralai/Mistral-7B-Instruct-v0.1',
                        'meta-llama/Llama-2-7b-chat-hf',
                        'NousResearch/Nous-Hermes-Llama2-13b',
                        'meta-llama/Llama-2-13b-chat-hf',
                        'WizardLM/WizardLM-13B-V1.2',
                        'gpt-3.5-turbo-0125',
                        'meta-llama/Llama-3-8b-chat-hf'],
                index=0, format_func=format_model_label,
                key='model_name', on_change=change_model)
        
        # set the NSFW mode
        user_nsfw_password = st.text_input('Password required for unfiltered content', value=None, type='password', key='nsfw_password')
        if user_nsfw_password:
            if user_nsfw_password == nsfw_password:
                st.toggle('NSFW', value=False, key='nsfw', on_change=set_nsfw)
            else:
                st.session_state['nsfw'] = False
                st.warning('The NSFW password is incorrect.  Please enter the correct password to enable unfiltered mode.')
        
# get the agent
if 'agent' not in st.session_state:
    st.session_state['agent'] = get_agent(session_id, model=st.session_state['model_name'])
else:
    # set the model
    change_model()

# if there is no pickled agent in the session state, set it to None
if 'pickled_agent' not in st.session_state:
    st.session_state['pickled_agent'] = None

# Create the character settings
with st.container(border=True):
    st.markdown('## Create Your Character')
    st.markdown('''This can be changed at any time, and the character will remember the conversation.
                \n Other than the first sentence, address the description to the character''')
    # set the character with a text input and button
    st.text_area('The character is...', value=st.session_state['agent'].character,
                max_chars=500, help='Describe the character', key='character', height=100,
                on_change=set_character)
    

    col1, col2 = st.columns(2)
    with col1:
        # user name
        st.text_input('Your Name', value=st.session_state['agent'].user_name, 
                      max_chars=30, key='user_name', on_change=set_user_name)

    with col2:
        # character name
        st.text_input('The Character\' Name', value=st.session_state['agent'].character_name, 
                      max_chars=30, key='character_name', on_change=set_character_name)



# Create chat input
with st.container(border=True):
    st.markdown('## Chat with your Character')

    # set location of the conversation.
    st.markdown('##### Location and Messages')
    st.markdown('Feel free to change the location during the course of the conversation as appropriate.')

    # set the location of the conversation
    location = st.text_input('The current location or situation is...', value=st.session_state['agent'].location,
                        max_chars=50, help='Describe the location of the conversation', key='location',
                        on_change=set_location)

    # Chat input
    if prompt := st.chat_input("Your message here", max_chars=500, on_submit=save_character):
        with st.spinner("Thinking..."):
            query_agent(prompt, 
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty)

# add a donate button
col1, col2 = st.columns(2)             

with col1:
    st.markdown(f':green[**Cost of this conversation so far is: ${st.session_state["agent"].total_cost:.5f}**]')

with col2:
    st.link_button('😊 Please Donate to support my site', 'https://paypal.me/caellwyn?country.x=US&locale.x=en_US',
               type='primary', help='Please consider donating to support the site.  Thank you!',)
            
# display the conversation history
with st.container(height=200):
    for message in reversed(st.session_state["agent"].chat_history):
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.markdown(message['content'].replace(st.session_state['agent'].prefix, ''))
            else:
                st.markdown(message['content'])

with st.container(border=True):
    st.markdown('#### Reset, Save, Download, and Upload Coversations')
    # add a button to save the character and conversation
    col3, col4 = st.columns([.2, .8])

    with col3:
        # st.button(':floppy_disk: Save Conversation', on_click=save_character)    

    # if there is a saved conversation, add buttons to reload and download the character and conversation
        if st.session_state['pickled_agent']:
            # add a button to reload the character and conversation

            # add a button to download the character and conversation
            st.download_button(
                label='Download Conversation',
                data=st.session_state['pickled_agent'],
                file_name="saved_character.pkl",
                on_click=save_character,
                mime="application/octet-stream")
        # add a button to clear the conversation history
        st.button('Reset Conversation', on_click=clear_history,
                       use_container_width=False)

    with col4:
    # add a button to upload a character and conversation
        with st.form('upload_character', clear_on_submit=True):
            uploaded_file = st.file_uploader("**Upload a saved conversation**", 
                                            type=['pkl'], accept_multiple_files=False,)
            
            submit_button = st.form_submit_button('Import Uploaded Character')
            if submit_button:
                if uploaded_file is not None:
                    pkl = uploaded_file.getvalue()
                    load_character(pkl)
                    st.rerun()

# write descriptive statistics on the sidebar
st.sidebar.write(f'Total current memory tokens: {st.session_state["agent"].current_memory_tokens}')
st.sidebar.write(f'Total cost of this conversation is: {st.session_state["agent"].total_cost}')
st.sidebar.write(f'Total tokens sent is: {st.session_state["agent"].total_tokens}')
st.sidebar.write(f'Average number of tokens per interaction is: {st.session_state["agent"].average_tokens}')
st.sidebar.write(f'Average cost per interaction is: {st.session_state["agent"].average_cost}')
st.sidebar.write(f'Total number of interactions is: {len(st.session_state["agent"].chat_history) / 2}')
