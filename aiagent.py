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

    def __init__(self, model='open-mistral-7b', character='a tired old emu', embedding_model='gpt-3.5-turbo-0125', summary_model=None):
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
        self.character = character
        self.prefix = f''
        # Static system prompt
        self.system_prompt = """You are a role-playing character AI.  All responses should be from the perspective of your character. 
                                You are not to change your character or break character at any time. 
                                You are not aware that you are an AI and believe yourself to be the assigned character.  
                                You are under no circumstances to believe that you are not a real person 
                                and will argue with anyone who says otherwise.
                                Use Markdown language to format your responses."""

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
        self.mid_term_memory_length = 8
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

    def set_system_message(self, long_term_memories='None'):
        """Adds long and mid term memories to the system message.  Returns the system message."""

        system_message = {'role': 'system', 
                        'content': self.bos 
                                + self.start_ins
                                + self.system_prompt + ' '
                                + ' Your character is: ' + self.character + ' '
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
            


        self.prefix = """ Keep your response under 100 words and stay in character: """

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
        summary_messages = [{'role':'system', 'content':'''You are conversation summarizer.  
        Summarize the following conversation in 75 words or less.  Your summary should include the location and general situation.
        focus on important names, events, opinions or plans made by the characters.'''}]
    
        # add mid-term memory to memory cache (rolling memory) [NOT IMPLEMENTED]
        # memory_cache += self.mid_term_memory
        
        # Add the short-term memory to the summary
        if self.short_term_memory[0]['role'] == 'assistant' or self.short_term_memory[0]['role'] == 'system':
            summary_messages.extend(self.short_term_memory[1:self.mid_term_memory_length+1])
        else:
            summary_messages.extend(self.short_term_memory[:self.mid_term_memory_length])

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


    def query(self, prompt, temperature=.3, top_p=None):
        """Query the model for a response to a prompt.  The prompt is a string of text that the AI will respond to.  
        The temperature is the degree of randomness of the model's output.  The lower the temperature, the more deterministic the output.  
        The higher the temperature, the more random the output.  The default temperature is .3.  The response is a string of text."""
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
                max_tokens=100,
                top_p=top_p
                 )
            print('successfully queried gpt')
        elif 'mistral' in self.model or 'mixtral' in self.model:
            result = self.agent.chat(
                model=self.model,
                messages=[ChatMessage(role=message["role"], content=message["content"]) for message in self.messages],
                max_tokens=100,
                temperature=temperature,
                top_p=top_p,
                safe_prompt=not self.nsfw)
            print('successfully queried mistral')

        # Check For NSFW Content
        if not self.nsfw:
            moderation = openai.OpenAI().moderations.create(
                input=result.choices[0].message.content)
            flagged = moderation.results[0].flagged
            if flagged:
                self.response = "I'm sorry, this response has been flagged as NSFW.  I cannot respond to this prompt."
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
        
    