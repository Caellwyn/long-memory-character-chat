import streamlit as st
import openai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import os, datetime
import pickle

# Load the OpenAI API key from the environment variables
try:
    with open(os.path.join(os.pardir,'mistral_ai_api.txt'), 'r') as f:
        mistral_api_key = f.read()
except Exception as e:
    mistral_api_key=st.secrets['MISTRAL_API_KEY']
    print(e)

try:
    with open(os.path.join(os.pardir,'chatgpt_api.txt'), 'r') as f:
        openai.api_key = f.read()

except:
    openai.api_key = st.secrets['OPENAI_API_KEY']


# model= 'gpt-3.5-turbo-0125'
# agent = openai.OpenAI()
# summary_agent = openai.OpenAI()
# summary_model = 'gpt-3.5-turbo-0125'

# model='open-mistral-7b'
# agent = MistralClient(api_key=api_key)
# summary_agent = MistralClient(api_key=mistral_api_key)
# summary_model = 'open-mistral-7b'
    
class Document:
    """Document class for storing text and metadata together.  This is used for storing long-term memory."""

    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

class AIAgent():
    """AIAgent class, which acts as the character to converse with
    Intialize with a character description, Defaults to: 'an attractive friend with a hidden crush'"""

    def __init__(self, model='open-mistral-7b', embedding_model='gpt', summary_model=None):
        # Initialize the AI agent
        self.set_model(model)

        # initialize the summary model
        if summary_model is None:
            summary_model = 'gpt-3.5-turbo-0125'
        self.set_summary_model(summary_model)

        # Initialize the embeddings model
        if 'gpt' in embedding_model:
            self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        elif 'mistral' in embedding_model:
            self.embeddings = MistralAIEmbeddings(api_key=mistral_api_key)

        # Set the character for the AI to role-play as
        self.character = 'A old emu with a tale to tell.  You desperately want someone to listen.'
        self.location = 'The Australian outback'
        self.user_name = 'User'
        self.character_name = 'Bill'
        self.prefix = f''
        
        # Static system prompt
        self.system_prompt = """You will be roleplaying as: {}.  Your name is {}.
        Speak from the perspective of this character using their voice, mannerisms, background knowledge, beliefs, and personality traits.
        
        Your current location is: {}. 
                
        Your responses should:
        - Use an informal, conversational tone with contractions, slang, rhetorical questions, figurative language, etc. 
        as appropriate for the character
        - Express desires, opinions, goals, and emotions fitting the character's persona
        - Make decisions and take actions the character would take, do not wait for the user to make decisions unless the character would.
        - Ask questions to learn more about topics or the user if relevant to the character.
        - Match the character's manner of speech and way of viewing the world and maintain a consistent tone and stayle and narrative flow.
        - Only include information and describe actions that the character would know or do based on their background.  If you don't know the answer to a question, truthfully say you do not know.
        - Remain fully in character throughout all responses.
        - Begin with "[{}]:" to indicate you are speaking as the provided character. Only use this tag once per response.
        - Be between 50 and 200 words, as is appropriate for the character's speaking style.
        - Use Markdown formatting like headers, italics and bold to enhance your response when appropriate. Do not use emojis or hashtags.
        
        Do not speak for the user or the AI, only the character you are roleplaying. Do not initiate any new prompts.
        You are to truly become this character and should not break character by referring to yourself as an AI, 
        acknowledging you are an AI assistant, or stating you are separate from the character you are portraying. 
        If prompted to do something out of character, provide an in-character response explaining why you would not do that.
        A good response will be engaging, entertaining, and consistent with the character's personality and background.
        
        You have a recent memory of: {}.  You also more distant memories of: {}.
        Only to refer to these memories if they are present and relevant, they might help you maintain a consistent persona and narrative."""

        # token and usage statistics
        self.total_cost = 0
        self.average_cost = 0
        self.total_tokens = 0
        self.average_tokens = 0
        self.current_memory_tokens = 0

        # string and instruction tokens (not currently used)
        self.bos = ''
        self.eos = ''
        self.start_ins = ''
        self.end_ins = ''

        # initialize the memory
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = 'nothing yet.'
        self.long_term_memories = 'nothing yet.'
        self.current_memory_id = 0
        self.messages = []
        self.system_message = self.set_system_message()

        # Set the short term memory length and overlap
        self.mid_term_memory_length = 4 ## must be even!
        if self.mid_term_memory_length % 2 != 0:
            self.mid_term_memory_length += 1
        self.mid_term_memory_overap = 2

        # NSFW filter
        self.nsfw = False

    def set_system_message(self):
        """Include dynamic elements in the system prompt.  Returns the system message."""

        self.system_message = {'role': 'system', 
                        'content': self.bos 
                                + self.start_ins
                                + self.system_prompt.format(self.character, 
                                                            self.character_name, 
                                                            self.location, 
                                                            self.character_name,
                                                            self.mid_term_memory,
                                                            self.long_term_memories)
                                + self.end_ins + ' '}
        

    def set_model(self, model='open-mistral-7b'):
        """Change the model the AI uses to generate responses.  Defaults to: 'open-mistral-7b'"""
        self.model = model
        if 'gpt' in self.model:
            self.agent = openai.OpenAI()
        elif 'mistral' in self.model or 'mixtral' in self.model:
            self.agent = MistralClient(api_key=mistral_api_key)

    def set_summary_model(self, summary_model='open-mistral-7b'):
        """Change the model the AI uses to summarize conversations.  Defaults to: 'open-mistral-7b'"""
        self.summary_model = summary_model
        if 'gpt' in self.summary_model:
            self.summary_agent = openai.OpenAI()
        elif 'mistral' in self.summary_model or 'mixtral' in self.summary_model:
            self.summary_agent = MistralClient(api_key=mistral_api_key)
    
    def set_character(self, character='a friendly old man.'):
        """Change the character the AI is role-playing as.  Defaults to: 'A friendly old man.'"""

        # Set the character for the AI to role-play as
        self.character = character
        self.system_message = self.set_system_message()

    def set_location(self, location='The Australian outback'):
        """Change the location the AI is role-playing in.  Defaults to: 'The Australian outback'"""

        # Set the location for the AI to role-play in
        self.location = location

    def set_user_name(self, user_name='User'):
        """Change the name the user is role-playing as.  Defaults to: 'User'"""

        # Set the user name for the AI to role-play as
        self.user_name = user_name

    def set_character_name(self, character_name='Character'):
        """Change the name of the AI's character.  Defaults to: 'Character'"""

        # Set the character name for the AI to role-play as
        self.character_name = character_name

    
    def add_message(self, text, role):
        """Adds a message to the AI's short term memory.  
        The message is a string of text and the role is either 'user' or 'assistant'."""
        
        self.chat_history.append({'role':role, 'content':text})

        # add a message to the AI's short term memory
        if role == 'user':
            self.short_term_memory.append({'role':role, 'content': self.start_ins 
                                           + f'<Location>: {self.location}, <Message>: ' 
                                           + text 
                                           + self.end_ins})
        if role == 'assistant':
            self.short_term_memory.append({'role':role, 'content': text + self.eos})

        # if the short-term memory is too long, summarize it, replace mid-term memory, and add it to the long term memory
            
        self.prefix = f""" Respond 50 - 100 words. Stay in character and be sure to maintain your personality and manner of speaking.: """

    def stringify_memory(self, memory):
        """Convert a memory list to a string.  Returns the string."""

        # convert a memory list to a string
        memory_string = ''
        for message in memory:
            memory_string += message['role'] + ': ' + message['content'] + ' '
        return memory_string

    def summarize_memories(self, max_tokens=200):
        """Summarize the short-term memory and add it to the mid-term memory.  
        Also add the mid-term memory to the long-term memory.  Returns nothing."""

        # Summary system message
        summary_prompt = {'role':'user', 'content':f'''
                          You are {self.character_name}'s conversation summarizer. 
                          Concisely summarize the key points covered in the converstion using the same speaking style, personality, as {self.character_name}.
                          The summary should capture:
                        - Important names, events, and be sure to mention your location.  If there is any indication of time or date, mention that as well.
                        - {self.character_name}'s and {self.user_name}'s opinions, feelings, attitude, or stance on topics discussed.
                        - Any significant changes or developments in your relationship with {self.user_name}.

                        Keep the summary brief, around 100-200 words. Speak authentically as {self.character_name}, 
                        using same informal voice you employ during conversations, including any accent, or quirks of your speech. 

                        The summary will be stored to reinforce {self.character_name}'s consistent persona and recollections over time. 
                        If asked to summarize something out-of-character, 
                        be sure to respond accordingly from {self.character_name}'s perspective.
                        '''}
    
        # add mid-term memory to memory cache (rolling memory) [NOT IMPLEMENTED]
        # memory_cache += self.mid_term_memory
        
        # Add the short-term memory to the summary, ens
        offset = 0
        while self.short_term_memory[offset]['role'] != 'user':
            offset += 1
        summary_messages = self.short_term_memory[offset:self.mid_term_memory_length + offset]
            
        summary_messages.append(summary_prompt)
        # Choose the model to use for summarization and summarize the conversation
        if 'gpt' in self.summary_model:
            summary = self.summary_agent.chat.completions.create(
                model=self.summary_model,
                messages=summary_messages, # this is the conversation history
                temperature=0, # this is the degree of randomness of the model's output
                max_tokens=max_tokens) 
        elif 'mistral' in self.summary_model or 'mixtral' in self.summary_model:
            messages=[ChatMessage(role=message["role"], content=message["content"]) for message in summary_messages]
            summary = self.summary_agent.chat(
                model=self.summary_model,
                messages=messages,
                max_tokens=max_tokens
                )
        print('LATEST SUMMARY: ', summary.choices[0].message.content)
        # add cost of message to total cost
        self.count_cost(summary, self.model)
        # add the current mid-term memory to the long-term memory
        if self.mid_term_memory != 'nothing':
            self.add_long_term_memory(self.mid_term_memory)
        # Store the summary as the new mid-term memory
        self.mid_term_memory = f"At {datetime.datetime.now().strftime('%H:%M:%S')}: {summary.choices[0].message.content}"

        # remove the oldest messages from the short-term memory

        self.short_term_memory = self.short_term_memory[-self.mid_term_memory_length + self.mid_term_memory_overap:]

    def add_long_term_memory(self, memory):
        """add a memory to the long-term memory vector store.  Returns nothing."""

        metadata = {'id':self.current_memory_id, 'timestamp':datetime.datetime.now()}
        self.current_memory_id += 1

        memory_document = Document(memory, metadata)
        # Use the OpenAIEmbeddings object for generating the embedding

        if not hasattr(self, 'long_term_memory_index'):

            self.long_term_memory_index = FAISS.from_documents([memory_document], self.embeddings)

        else: 
            self.long_term_memory_index.add_documents([memory_document], encoder=self.embeddings)


    def query_long_term_memory(self, query, k=2):
        """Query the long-term memory for similar documents.  Returns a list of Document objects."""
        try:
            memories = self.long_term_memory_index.amax_marginal_relevance_search(query, k=k, fetch_k = k*3)
            sorted_memories = sorted(memories, key=lambda x: x.metadata['timestamp'])
        except Exception as e:
            print('no memories found')
            print(e)
            return []

        return sorted_memories

    def count_cost(self, result, model):
        """Count the cost of the messages.  
        The cost is calculated as the number of tokens in the input and output times the cost per token.  
        Returns the cost."""

        # cost is calculated as the number of tokens in the input and output times the cost per token
        if model.startswith('gpt-3'):
            input_cost = 0.0005 / 1000
            output_cost = 0.0015 / 1000
        elif model.startswith('gpt-4'):
            input_cost = 0.01 / 1000
            output_cost = 0.03 / 1000
        elif model.startswith('open-mistral-7b'):
            input_cost = 0.00025 / 1000
            output_cost = 0.00025 / 1000
        elif model.startswith('open-mixtral-8x7b'):
            input_cost = 0.0007 / 1000
            output_cost = 0.0007 / 1000
        elif model.startswith('text-embedding-3-small'):
            input_cost = 0.000002 / 1000
            output_cost = 0.000002 / 1000
        elif model.startswith('text-embedding-3-large'):
            input_cost = 0.00013 / 1000
            output_cost = 0.00013 / 1000
        else:
            print('Model not recognized')
        
        # determine the length of inputs and outputs
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens
        self.current_memory_tokens = input_tokens + output_tokens
        lastest_cost = input_cost * input_tokens + output_cost * output_tokens

        self.total_tokens += self.current_memory_tokens
        self.total_cost += lastest_cost
        self.average_cost = self.total_cost / (len(self.chat_history) / 2)
        self.average_tokens = self.total_tokens / (len(self.chat_history) / 2) 

        # calculate the cost
        return lastest_cost


    def query(self, prompt, temperature=.3, top_p=None, max_tokens=100):
        """Query the model for a response to a prompt.  The prompt is a string of text that the AI will respond to.  
        The temperature is the degree of randomness of the model's output.  The lower the temperature, the more deterministic the output.  
        The higher the temperature, the more random the output.  The default temperature is .3.  The response is a string of text."""
        
        prompt = f'[{self.user_name}]: {prompt} '
        self.messages = []
        # build the full model prompt
        # Query the long-term memory for similar documents
        if hasattr(self, 'long_term_memory_index'):
            returned_memories = self.query_long_term_memory(prompt)
            if len(returned_memories) > 0:
                # convert the memories to a string
                retrieved_memories = {doc.page_content for doc in returned_memories} # Remove duplicate memories
                self.long_term_memories = ' : '.join(retrieved_memories)
                for i, memory in enumerate(retrieved_memories):
                    print(f"<<Vector DB retrieved Memory {i}>>:\n", memory)
            else:
                print('no memories retrieved')
        else:
            print('no memories yet')

        self.set_system_message()

        self.messages.append(self.system_message)
        self.messages.extend(self.short_term_memory)
        self.messages.append({'role':'user', 'content': self.start_ins + self.prefix + prompt + self.end_ins})

        # Query the model through the API
        if 'gpt' in self.model:
            result = self.agent.chat.completions.create(
                model=self.model,
                messages=self.messages, # this is the conversation history
                temperature=temperature, # this is the degree of randomness of the model's output
                max_tokens=max_tokens,
                top_p=top_p
                 )
        elif 'mistral' in self.model or 'mixtral' in self.model:
            result = self.agent.chat(
                model=self.model,
                messages=[ChatMessage(role=message["role"], content=message["content"]) for message in self.messages],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                safe_prompt=not self.nsfw)

        # Check For NSFW Content
        if not self.nsfw:
            moderation = openai.OpenAI().moderations.create(
                input=result.choices[0].message.content)
            flagged = moderation.results[0].flagged
            if flagged:
                self.response = "[System]: I'm sorry, this response has been flagged as NSFW and cannot be shown."
                print('NSFW content detected')
            else:
                self.response = result.choices[0].message.content
                print('No NSFW content detected')
        else:
            # Store the response
            self.response = result.choices[0].message.content
        
        # add response to current message history
        self.messages.append({'role':'assistant', 'content':self.response})

        # Add user prompt to message history
        self.add_message(prompt, role='user')
        
        # Add reply to message history
        self.add_message(self.response, role='assistant')

        # add cost of message to total cost
        self.count_cost(result, self.model)

        if len(self.short_term_memory) > self.mid_term_memory_length * 2:
            self.summarize_memories()

        return self.response
    
    def clear_history(self):
        """Clear the AI's memory.  Returns nothing."""
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = 'nothing, yet'
        self.long_term_memory = 'nothing, yet'
        self.current_memory_id = 0
        self.prefix = ''
        self.response = "I'm thinking of my response"
        self.system_message = self.set_system_message()
        self.messages = []
        self.total_cost = 0
        self.total_tokens = 0
        self.current_memory_tokens = 0
        self.average_tokens = 0
        if hasattr(self, 'long_term_memory_index'):
            del self.long_term_memory_index
        
    def get_memory(self):
        """Return the AI's current memory.  Returns a list of messages."""
        return self.messages
    
    def get_history(self):
        """Return the AI's full chat history.  Returns a list of messages."""
        return self.chat_history
    
    def save_agent(self):
        """Save agent to path"""
        saved_attrs = self.__dict__.copy()
        del saved_attrs['summary_agent']
        del saved_attrs['agent']
        del saved_attrs['embeddings']
        if 'long_term_memory_index' in saved_attrs.keys():
            saved_attrs['long_term_memory_index'] = saved_attrs['long_term_memory_index'].serialize_to_bytes()
        return pickle.dumps(saved_attrs)

    def load_agent(self, file):
        """Load saved agent from path"""
        loaded_attrs = pickle.loads(file)

        # Replace agent attributes with loaded attributes
        for key, value in loaded_attrs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

        self.set_summary_model(self.summary_model)
        self.set_model(self.model)

        # De-serialize the vector db
        if 'long_term_memory_index' in loaded_attrs:
            self.long_term_memory_index = FAISS.deserialize_from_bytes(loaded_attrs['long_term_memory_index'], 
                                                                                   self.embeddings)
            print('deserialized long term memory index')
        
    