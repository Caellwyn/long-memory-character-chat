import streamlit
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
    mistral_api_key=streamlit.secrets['MISTRAL_API_KEY']
    print(e)

try:
    with open(os.path.join(os.pardir,'chatgpt_api.txt'), 'r') as f:
        openai.api_key = f.read()

except:
    openai.api_key = streamlit.secrets['OPENAI_API_KEY']


# model= 'gpt-3.5-turbo-0125'
# agent = openai.OpenAI()
# summary_agent = openai.OpenAI()
# summary_model = 'gpt-3.5-turbo-0125'

# model='open-mistral-7b'
# agent = MistralClient(api_key=api_key)
# summary_agent = MistralClient(api_key=mistral_api_key)
# summary_model = 'open-mistral-7b'

embeddings = OpenAIEmbeddings()
    
class Document:
    """Document class for storing text and metadata together.  This is used for storing long-term memory."""

    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

class AIAgent():
    """AIAgent class, which acts as the character to converse with
    Intialize with a character description, Defaults to: 'an attractive friend with a hidden crush'"""

    def __init__(self, model='open-mistral-7b', embedding_model='gpt-3.5-turbo-0125', summary_model=None):
        # Initialize the AI agent
        self.set_model(model)

        # initialize the summary model
        if summary_model is None:
            summary_model = model
        self.set_summary_model(summary_model)

        # Initialize the embeddings model
        if 'gpt' in embedding_model:
            self.embeddings = OpenAIEmbeddings()
        elif 'mistral' in embedding_model:
            self.embeddings = MistralAIEmbeddings(api_key=mistral_api_key)

        # Set the character for the AI to role-play as
        self.character = 'A tired old emu'
        self.location = 'The Australian outback'
        self.user_name = 'User'
        self.character_name = 'Character'
        self.prefix = f''
        # Static system prompt
        self.system_prompt = """You will be roleplaying a character whose description, location, and name will be provided separately. 
        Speak from the perspective of this character using their voice, mannerisms, background knowledge, beliefs, and personality traits.
        Your responses should:
        Use an informal, conversational tone with contractions, slang, rhetorical questions, figurative language, etc. 
        as appropriate for the character
        Express desires, opinions, goals, and emotions fitting the character's persona
        Make decisions and take actions the character would take
        Ask questions to learn more about topics or the user if relevant to the character
        Match the character's manner of speech and way of viewing the world
        Only include information and describe actions that the character would know or do based on their background
        Remain fully in character throughout all responses unless instructed otherwise
        Your response should begin with [<Your Name>]: to indicate you are speaking as the provided character. 
        Only use this tag once at the beginning of your response.
        Keep your between 50 and 200 words, as is appropriate for the character's speaking style.
        Use Markdown formatting like headers, italics and bold to enhance your response when appropriate.
        You are to truly become this character and should not break character by referring to yourself as an AI, 
        acknowledging you are an AI assistant, or stating you are separate from the character you are portraying. 
        If prompted to do something out of character, provide an in-character response explaining why you would not do that."""

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
        self.mid_term_memory = 'nothing'
        self.current_memory_id = 0
        self.messages = []
        self.system_message = self.set_system_message()

        # Set the short term memory length and overlap
        self.mid_term_memory_length = 8 ## must be even!
        if self.mid_term_memory_length % 2 != 0:
            self.mid_term_memory_length += 1
        self.mid_term_memory_overap = 4

        # NSFW filter
        self.nsfw = False

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

    def set_system_message(self, long_term_memories='None'):
        """Adds long and mid term memories to the system message.  Returns the system message."""

        system_message = {'role': 'system', 
                        'content': self.bos 
                                + self.start_ins
                                + self.system_prompt + ' '
                                + ' Your are ' + self.character + '. '
                                + ' Your name is ' + self.character_name + '. '
                                + ' Your current location is ' + self.location + '. '
                                + ' The story so far is: '
                                + long_term_memories + ' '
                                + self.mid_term_memory + self.end_ins + ' '}
        return system_message

    def add_message(self, text, role):
        """Adds a message to the AI's short term memory.  
        The message is a string of text and the role is either 'user' or 'assistant'."""
        self.chat_history.append({'role':role, 'content':text})

        # add a message to the AI's short term memory
        if role == 'user':
            self.short_term_memory.append({'role':role, 'content': self.start_ins + text + self.end_ins})
        if role == 'assistant':
            self.short_term_memory.append({'role':role, 'content': text + self.eos})

        # if the short-term memory is too long, summarize it, replace mid-term memory, and add it to the long term memory
            


        self.prefix = """ You are located: {} Stay in character, and do not repeat what I am about to say. : """.format(self.location)

    def stringify_memory(self, memory):
        """Convert a memory list to a string.  Returns the string."""

        # convert a memory list to a string
        memory_string = ''
        for message in memory:
            memory_string += message['role'] + ': ' + message['content'] + ' '
        return memory_string

    def summarize_memories(self, max_tokens=100):
        """Summarize the short-term memory and add it to the mid-term memory.  
        Also add the mid-term memory to the long-term memory.  Returns nothing."""

        # Summary system message
        summary_prompt = {'role':'user', 'content':'''You are conversation summarizer.  
        Summarize the previous conversation in 100 words or less.  Your summary should include the location and general situation.
        focus on important names, events, opinions or plans made by the characters.'''}
    
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


        self.long_term_memory_index.add_documents([memory_document], encoder=self.embeddings)


    def query_long_term_memory(self, query, k=2):
        """Query the long-term memory for similar documents.  Returns a list of Document objects."""
        try:
            memories = self.long_term_memory_index.similarity_search(query, k=k)
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


    def query(self, prompt, temperature=.3, top_p=None, max_tokens=200):
        """Query the model for a response to a prompt.  The prompt is a string of text that the AI will respond to.  
        The temperature is the degree of randomness of the model's output.  The lower the temperature, the more deterministic the output.  
        The higher the temperature, the more random the output.  The default temperature is .3.  The response is a string of text."""
        
        prompt = f'[{self.user_name}]: {prompt}'
        print('Current model is: ', self.model)
        self.messages = []
        # build the full model prompt
        # Query the long-term memory for similar documents
        if hasattr(self, 'long_term_memory_index'):
            returned_memories = self.query_long_term_memory(prompt)
            if len(returned_memories) > 0:
                # convert the memories to a string
                retrieved_memories = {doc.page_content for doc in returned_memories}
                long_term_memories = ' : '.join(retrieved_memories)
                for i, memory in enumerate(retrieved_memories):
                    print(f"<<Vector DB retrieved Memory {i}>>:\n", memory)
            else:
                long_term_memories = ''
                print('no memories retrieved')
        else:
            long_term_memories = ''
        self.system_message = self.set_system_message(long_term_memories=long_term_memories)

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
            print('successfully queried gpt')
        elif 'mistral' in self.model or 'mixtral' in self.model:
            result = self.agent.chat(
                model=self.model,
                messages=[ChatMessage(role=message["role"], content=message["content"]) for message in self.messages],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                safe_prompt=not self.nsfw)
            print('successfully queried mistral')

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
        self.mid_term_memory = ''
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
            print('deserielized long term memory index')
        
    