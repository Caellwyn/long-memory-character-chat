import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os, datetime
import streamlit
import pickle

# Load the OpenAI API key from the environment variables
try:
    with open(os.path.join(os.pardir,'chatgpt_api.txt'), 'r') as f:
        openai.api_key = f.read()

except:
    openai.api_key = streamlit.secrets['OPENAI_API_KEY']

agent = openai.OpenAI()
embeddings = OpenAIEmbeddings()
    
class Document:
    """Document class for storing text and metadata together.  This is used for storing long-term memory."""

    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

class AIAgent():
    """AIAgent class, which acts as the character to converse with
    Intialize with a character description, Defaults to: 'A friendly old man.'"""

    def __init__(self, character='a friendly old man', agent=openai.OpenAI(), embeddings = OpenAIEmbeddings(), chat_model='gpt-3.5-turbo-0125', ):
        # Initialize the AI agent with the OpenAI API
        self.agent = agent
        self.model = chat_model
        self.character = character
        # prefix for user messages after the first message
        self.prefix = f'Do not change your persona: '
        # Static system prompt
        self.system_prompt = """You are a role-playing character AI.  All responses should be from the perspective of your character. 
                                You are not to change your character or break character at any time. 
                                You are not aware that you are an AI and believe yourself to be the assigned character.  
                                You are under no circumstances to believe that you are not a real person 
                                and will argue with anyone who says otherwise."""
        # Initialize the AI's memory and other trackers
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = ''
        self.embeddings = embeddings
        self.current_memory_id = 0
        self.prefix = ''
        self.response = "I'm thinking of my response"
        self.system_message = self.set_system_message()
        self.messages = []
        self.total_cost = 0
        self.average_cost = 0
        self.total_tokens = 0
        self.average_tokens = 0
        self.current_memory_tokens = 0
    
    def set_character(self, character='a friendly old man.'):
        """Change the character the AI is role-playing as.  Defaults to: 'A friendly old man.'"""

        # Set the character for the AI to role-play as
        self.character = character
        self.system_message = self.set_system_message()

    def set_system_message(self, long_term_memories=''):
        """Adds long and mid term memories to the system message.  Returns the system message."""

        system_message = {'role': 'system', 
                        'content': self.system_prompt
                                + ' Your character is: ' + self.character
                                + ' The story so far is: '
                                + long_term_memories + ' '
                                + self.mid_term_memory + ' '}
        return system_message

    def add_message(self, text, role):
        """Adds a message to the AI's short term memory.  
        The message is a string of text and the role is either 'user' or 'assistant'."""

        # add a message to the AI's short term memory
        message = {'role':role, 'content':text}
        self.short_term_memory.append(message)
        self.chat_history.append(message)

        # if the short-term memory is too long, summarize it, replace mid-term memory, and add it to the long term memory
        if len(self.short_term_memory) > 10:
            self.summarize_memories()

    def stringify_memory(self, memory):
        """Convert a memory list to a string.  Returns the string."""

        # convert a memory list to a string
        memory_string = ''
        for message in memory:
            memory_string += message['role'] + ' ' + message['content'] + ' '
        return memory_string

    def summarize_memories(self, max_tokens=100):
        """Summarize the short-term memory and add it to the mid-term memory.  
        Also add the mid-term memory to the long-term memory.  Returns nothing."""

        # summarize the memory cache
        summary_prompt = '''You are conversation summarizer.  Summarize the following conversation to extract key details as well as the overall 
        situation.  The conversation is as follows: '''
        # add oldest messages to memory cache
        memory_cache = self.stringify_memory(self.short_term_memory[:5])
        # add mid-term memory to memory cache (rolling memory) [NOT IMPLEMENTED]
        # memory_cache += self.mid_term_memory

        # summarize the memory cache
        summary_messages = [{'role':'user','content':summary_prompt + memory_cache}]
        summary = self.agent.chat.completions.create(
            model=self.model,
            messages=summary_messages, # this is the conversation history
            temperature=.1,
            max_tokens=max_tokens) # this is the degree of randomness of the model's output
        print('<<SUMMARY>>:', summary.choices[0].message.content) ## REMOVE        
        # add cost of message to total cost
        self.total_cost += self.count_cost(summary, self.model)

        # add the current mid-term memory to the long-term memory
        self.add_long_term_memory(self.mid_term_memory)
        
        # Store the summary as the new mid-term memory
        self.mid_term_memory = summary.choices[0].message.content

        # remove the oldest messages from the short-term memory
        self.short_term_memory = self.short_term_memory[5:]

    def add_long_term_memory(self, memory):
        """add a memory to the long-term memory vector store.  Returns nothing."""

        metadata = {'id':self.current_memory_id, 'timestamp':datetime.datetime.now()}
        self.current_memory_id += 1
        memory_document = Document(memory, metadata)
        # Use the OpenAIEmbeddings object for generating the embedding

        if not hasattr(self, 'long_term_memory_index'):
            self.long_term_memory_index = FAISS.from_documents([memory_document], self.embeddings)
        self.long_term_memory_index.add_documents([memory_document], encoder=self.embeddings)

    def query_long_term_memory(self, query, k=3):
        """Query the long-term memory for similar documents.  Returns a list of Document objects."""
        return self.long_term_memory_index.similarity_search(query, k=k)

    def count_cost(self, result, model):
        """Count the cost of the messages.  
        The cost is calculated as the number of tokens in the input and output times the cost per token.  
        Returns the cost."""

        # cost is calculated as the number of tokens in the input and output times the cost per token
        if model.startswith('gpt-3'):
            input_cost = .0005 / 1000
            output_cost = .0015 / 1000
        elif model.startswith('gpt-4'):
            input_cost = .01 / 1000
            output_cost = .03 / 1000
        
        # determine the length of inputs and outputs
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens
        new_tokens = input_tokens + output_tokens
        self.total_tokens += new_tokens

        if self.average_tokens > 0:
            self.average_tokens = (self.average_tokens + new_tokens) / 2
        else:
            self.average_tokens = new_tokens

        self.current_memory_tokens = input_tokens + output_tokens
        self.total_cost += input_cost * input_tokens + output_cost * output_tokens
        if self.average_cost > 0:
            self.average_cost = (self.average_cost + input_cost * input_tokens + output_cost * output_tokens) / 2
        else:
            self.average_cost = input_cost * input_tokens + output_cost * output_tokens

        # calculate the cost
        return input_cost * input_tokens + output_cost * output_tokens


    def query(self, prompt, temperature=.3):
        """Query the model for a response to a prompt.  The prompt is a string of text that the AI will respond to.  
        The temperature is the degree of randomness of the model's output.  The lower the temperature, the more deterministic the output.  
        The higher the temperature, the more random the output.  The default temperature is .3.  The response is a string of text."""

        self.messages = []
        # build the full model prompt
        if hasattr(self, 'long_term_memory_index'):
            retrieved_memories = [doc.page_content for doc in self.query_long_term_memory(prompt)]
            long_term_memories = ' '.join(retrieved_memories)
            for i, memory in enumerate(retrieved_memories):
                print(f"<<Vector DB retrieved Memory {i}>>:\n", memory)
        else:
            long_term_memories = ''
        self.system_message = self.set_system_message(long_term_memories=long_term_memories)
        self.messages.append(self.system_message)
        self.messages.extend(self.short_term_memory)
        self.messages.append({'role':'user', 'content':self.prefix + prompt})

        # Query the model through the API
        result = self.agent.chat.completions.create(
            model=self.model,
            messages=self.messages, # this is the conversation history
            temperature=temperature, # this is the degree of randomness of the model's output
            max_tokens=300
        )
        # Store the response
        self.response = result.choices[0].message.content

        # add response to current message history
        self.messages.append({'role':'assistant', 'content':self.response})
        # add cost of message to total cost
        self.count_cost(result, self.model)

        

        # Add user prompt to message history
        self.add_message(prompt, role='user')
        
        # Add reply to message history
        self.add_message(self.response, role='assistant')

        
        return result.choices[0].message.content
    
    def clear_history(self):
        """Clear the AI's memory.  Returns nothing."""
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = ''
        self.embeddings = OpenAIEmbeddings()
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
        del saved_attrs['agent']
        del saved_attrs['embeddings']
        if 'long_term_memory_index' in saved_attrs.keys():
            saved_attrs['long_term_memory_index'] = saved_attrs['long_term_memory_index'].serialize_to_bytes()
        return pickle.dumps(saved_attrs)

    def load_agent(self, file, embeddings=None, agent=None):
        """Load saved agent from path"""
        loaded_attrs = pickle.loads(file)

        # Use either current agent embeddings and agent or specified agent and embeddings.  These cannot be serialized.
        if not agent:
            loaded_attrs['agent'] = self.agent
        if not embeddings:
            loaded_attrs['embeddings'] = self.embeddings
        # Replace agent attributes with loaded attributes
        for key, value in loaded_attrs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

        # De-serialize the vector db
        if 'long_term_memory_index' in loaded_attrs:
            self.long_term_memory_index = FAISS.deserialize_from_bytes(loaded_attrs['long_term_memory_index'], 
                                                                                   self.embeddings)
        
    