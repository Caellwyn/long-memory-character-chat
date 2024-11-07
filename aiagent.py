import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os, datetime, pickle, time
import google.generativeai as genai
import anthropic
from google.generativeai.types import HarmCategory, HarmBlockThreshold

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
except:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)


class Document:
    """Document class for storing text and metadata together.  This is used for storing long-term memory."""

    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class AIAgent:
    """AIAgent class, which acts as the character to converse with
    Intialize with a character description, Defaults to: 'an attractive friend with a hidden crush'
    """

    def __init__(
        self,
        model="open-mistral-7b",
        embedding_model="gpt",
        summary_model="gemini-1.5-flash",
    ):
        # Initialize the AI agent
        self.set_model(model)

        # initialize the summary model
        self.set_summary_model(summary_model)

        # Initialize the embeddings model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Set the character for the AI to role-play as
        self.character = (
            "A old emu with a tale to tell.  You desperately want someone to listen."
        )
        self.location = "The Australian Outback"
        self.user_name = "User"
        self.character_name = "Bill"
        self.prefix = ""

        # token and usage statistics
        self.total_cost = 0
        self.average_cost = 0
        self.total_tokens = 0
        self.average_tokens = 0
        self.current_memory_tokens = 0

        # string and instruction tokens (not currently used)
        self.bos = ""
        self.eos = ""
        self.start_ins = ""
        self.end_ins = ""

        # initialize the memory
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = "nothing yet."
        self.long_term_memories = "nothing yet."
        self.current_memory_id = 0
        self.messages = []
        self.message_style_sample = None
        self.response = "I'm thinking of my response"
        # Set the system prompt to instruct the AI on how to role-play
        self.system_message = self.set_system_message()

        # Set the short term memory length and overlap

        ## How many messages are summarized:
        ## must be even!
        self.mid_term_memory_length = 8

        ## How long short-term memory can grow:
        ## must be greater than mid_term_memory_length
        ## must be even
        self.max_short_term_memory_length = 16

        ## How much overlap between each summarized mid-term memory:
        ## must be less than mid_term_memory_length
        ## must be Even
        self.mid_term_memory_overlap = 2

        ## Checks to enforce length rules
        if self.mid_term_memory_length > self.max_short_term_memory_length:
            self.max_short_term_memory_length = self.mid_term_memory_length + 2

        if self.mid_term_memory_overlap > self.mid_term_memory_length:
            self.mid_term_memory_overlap = self.mid_term_memory_length - 2

        ## Checks to enforce length rules
        if self.mid_term_memory_length % 2 != 0:
            self.mid_term_memory_length += 1

        if self.max_short_term_memory_length % 2 != 0:
            self.max_short_term_memory_length += 1

        if self.mid_term_memory_overlap % 2 != 0:
            self.mid_term_memory_overlap += 1

        # NSFW filter
        self.nsfw = False

    def set_system_message(self) -> None:
        """Include dynamic elements in the system prompt.  Returns the system message."""

        # Set the system prompt to instruct the AI on how to role-play
        self.system_prompt = f"""
        Roleplay as {self.character}, named {self.character_name}. 
        {self.message_style_sample}
        Fully embody this character's personality, voice, mannerisms, knowledge, beliefs, and traits. The current situation is: {self.location}.
        Respond in informal, conversational language using dialogue, body language (asterisks) and actions to show the character's desires, opinions, goals and emotions.
        Do not respond with emotions directly, but show them through the character's actions, expression, body language, and dialogue.
        Proactively make decisions and advance the plot. Ask questions to learn more, when relevant.
        Maintain consistent speech patterns, worldview and only include details the character would know. Begin responses with '[{self.character_name}]:'.
        Keep responses 50-120 words. Use markdown formatting (italics for actions, bold for emphasis) as suitable.
        Carefully examine the following notes for relevant information.  These are summarized memories for your character.
        If any information is especially relevant to the conversation, feel free to mention it or use it as implicit context for your responses as would be appropriate.
        Recent notes: {self.mid_term_memory} 
        Notes from longer ago: {self.long_term_memories}
        Only answer questions with information your character would know.  
        If you are asked about previous events relating to your history specifically with the user, check your notes for the answer.  
        If the information is not in your notes, ask the user to tell you some details to help you remember.  
        If you have already asked the user to help you remember, and the information is still not in your notes, then say you don't remember.
        The goal is an immersive, consistent roleplaying experience where the user feels a sense of narrative progress and connection to the dynamic, engaging character.
        """

        self.system_message = {
            "role": "system",
            "content": self.bos
            + self.start_ins
            + self.system_prompt
            + self.end_ins
            + " ",
        }

    def set_model(self, model="gpt-3.5-turbo-0125") -> None:
        """Change the model the AI uses to generate responses.  Defaults to: 'open-mistral-7b'"""
        self.model = model
        if "gpt" in self.model:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
            except:
                api_key = st.secrets["OPENAI_API_KEY"]
            self.agent = openai.OpenAI(
                api_key=api_key, base_url="https://api.openai.com/v1"
            )

        elif "gemini" in self.model:
            self.agent = genai.GenerativeModel(model_name=self.model)

        elif "claude" in self.model:
            try:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            except:
                try:
                    api_key = st.secrets["ANTHROPIC_API_KEY"]
                except Exception as e:
                    print(e)

            self.agent = anthropic.Anthropic(api_key=api_key)

        elif "hermes" in self.model:
            try:
                api_key = os.getenv("LAMBDA_API_KEY")
            except:
                try:
                    api_key = st.secrets("LAMBDA_API_KEY")
                except Exception as e:
                    print(e)
            self.agent = openai.OpenAI(
                api_key=api_key, base_url="https://api.lambdalabs.com/v1"
            )

        else:
            try:
                api_key = os.getenv("TOGETHER_API_KEY")
            except:
                api_key = st.secrets["TOGETHER_API_KEY"]
            self.agent = openai.OpenAI(
                api_key=api_key, base_url="https://api.together.xyz/v1"
            )

    def set_summary_model(self, summary_model="gpt-3.5-turbo-0125") -> None:
        """Change the model the AI uses to summarize conversations.  Defaults to: 'open-mistral-7b'"""
        self.summary_model = summary_model
        if "gpt" in self.summary_model:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
            except:
                api_key = st.secrets["OPENAI_API_KEY"]
            self.summary_agent = openai.OpenAI(
                api_key=api_key, base_url="https://api.openai.com/v1"
            )

        elif "gemini" in self.summary_model:
            self.summary_agent = genai.GenerativeModel(model_name=self.summary_model)

        elif "claude" in self.summary_model:
            try:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            except:
                try:
                    api_key = st.secrets["ANTHROPIC_API_KEY"]
                except Exception as e:
                    print(e)

            self.summary_agent = anthropic.Anthropic(api_key=api_key)

        else:
            try:
                api_key = os.getenv("TOGETHER_API_KEY")
            except:
                api_key = st.secrets["TOGETHER_API_KEY"]
            self.summary_agent = openai.OpenAI(
                api_key=api_key, base_url="https://api.together.xyz/v1"
            )

    def set_character(self, character="a friendly old man.") -> None:
        """Change the character the AI is role-playing as.  Defaults to: 'A friendly old man.'"""

        # Set the character for the AI to role-play as
        self.character = character

    def set_location(self, location="The Australian outback") -> None:
        """Change the location the AI is role-playing in.  Defaults to: 'The Australian outback'"""

        # Set the location for the AI to role-play in
        self.location = location

    def set_user_name(self, user_name="User") -> None:
        """Change the name the user is role-playing as.  Defaults to: 'User'"""

        # Set the user name for the AI to role-play as
        self.user_name = user_name

    def set_character_name(self, character_name="Character") -> None:
        """Change the name of the AI's character.  Defaults to: 'Character'"""

        # Set the character name for the AI to role-play as
        self.character_name = character_name

    def add_message(self, text, role) -> None:
        """Adds a message to the AI's short term memory.
        The message is a string of text and the role is either 'user' or 'assistant'."""

        self.chat_history.append({"role": role, "content": text})

        # add a message to the AI's short term memory
        if role == "user":
            self.short_term_memory.append(
                {
                    "role": role,
                    "content": self.start_ins
                    + f"<Location>: {self.location}, <Message>: "
                    + text
                    + self.end_ins,
                }
            )
        if role == "assistant":
            self.short_term_memory.append({"role": role, "content": text + self.eos})

        # if the short-term memory is too long, summarize it, replace mid-term memory, and add it to the long term memory

        self.prefix = f""" Do not repeat phrases from your most recent response: """

    def summarize_memories(self, max_tokens=150, temperature=0, top_p=0.05) -> None:
        """Summarize the short-term memory and add it to the mid-term memory.
        Also add the mid-term memory to the long-term memory.  Returns nothing."""

        # Summary system message
        summary_prompt = {
            "role": "user",
            "content": f"""You are {self.character_name}'s memory creator.  You will create a summary to help them remember the conversation up to this point.
                        It's your job to help {self.character_name} remember their current situation and any important events that have occurred for later reference.
                        Your summary will serve as both a short-term and long-term memory for {self.character_name}.
                        Anything you do not include in your summary will be forgotten by {self.character_name}.
                        Describe the current situation and the context of the discussion, including location, date and time if possible, and any notable events.
                        Also record {self.character_name}'s and {self.user_name}'s opinions or feelings about topics discussed, changes in their relationship, and any important decisions made.
                        Be sure to keep track of any important details or information that could important later, for example names and information about other characters.
                        Your most recent memory is here in backticks:`{self.mid_term_memory}`
                        Some other past memories that recently came to mind are here in double backticks:``{self.long_term_memories}``
                        Use these memories to create context for the new memory you are creating.
                        Do not put anything in your response that is not already in the provided conversation.
                        Your response should be no more than 125 words.""",
        }

        # Add the short-term memory to the summary, ensures a 'user' role message is first.
        offset = 0
        for memory in self.short_term_memory:
            if memory["role"] == "user":
                break
            else:
                offset += 1

        # add the most recent conversation to the summary
        summary_messages = self.short_term_memory[
            offset : self.mid_term_memory_length + offset
        ]
        summary_messages.append(summary_prompt)

        # Save original summary model name in case we need to use a backup model (Because Gemini models are not always available)
        original_summary_model = self.summary_model

        # Choose the model to use for summarization and summarize the conversation
        if "gemini" in self.summary_model:
            try:

                config = genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                )
                if self.nsfw:
                    safety_settings = {
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                else:
                    safety_settings = None
                # format the messages for the Gemini model
                gemini_summary_messages = self.format_messages_for_gemini(
                    summary_messages
                )

                response = self.summary_agent.generate_content(
                    generation_config=config,
                    contents=gemini_summary_messages,
                    safety_settings=safety_settings,
                )
                summary = response.text

            except:
                print(
                    "Gemini model failed to load.  Using GPT-3.5-turbo-0125 model for summary"
                )

                self.set_summary_model("gpt-4o-mini")

                response = self.summary_agent.chat.completions.create(
                    model=self.summary_model,
                    messages=summary_messages,  # this is the conversation history
                    temperature=temperature,  # this is the degree of randomness of the model's output
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                summary = response.choices[0].message.content

        elif "claude" in self.summary_model:
            # format the messages for the Claude model (He doesn't like trailing spaces)
            summary_messages[-1]["content"] = summary_messages[-1]["content"].strip()

            # Query Claude for a Summary
            response = self.summary_agent.messages.create(
                model=self.summary_model,
                messages=summary_messages,  # this is the conversation history
                temperature=temperature,  # this is the degree of randomness of the model's output
                max_tokens=max_tokens,
                top_p=top_p,
            )
            summary = response.content[0].text

        else:
            # Try to use the OpenAI API for summary

            response = self.summary_agent.chat.completions.create(
                model=self.summary_model,
                messages=summary_messages,  # this is the conversation history
                temperature=temperature,  # this is the degree of randomness of the model's output
                max_tokens=max_tokens,
                top_p=top_p,
            )
            summary = response.choices[0].message.content

        print(f"LATEST SUMMARY: {summary}")
        # add cost of message to total cost
        self.count_cost(response, self.summary_model, summary=True)
        # add the current mid-term memory to the long-term memory
        if self.mid_term_memory != "nothing yet.":
            self.add_long_term_memory(self.mid_term_memory)
        # Store the summary as the new mid-term memory
        self.mid_term_memory = (
            f"At {datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}: {summary}"
        )

        # remove the oldest messages from the short-term memory

        self.short_term_memory = self.short_term_memory[
            (offset + self.mid_term_memory_length) - self.mid_term_memory_overlap :
        ]


        if not self.summary_model == original_summary_model:
            self.set_summary_model(original_summary_model)

    def add_long_term_memory(self, memory) -> None:
        """add a memory to the long-term memory vector store.  Returns nothing."""

        metadata = {
            "id": self.current_memory_id,
        }
        self.current_memory_id += 1

        memory_document = Document(memory, metadata)
        # Use the OpenAIEmbeddings object for generating the embedding

        if not hasattr(self, "long_term_memory_index"):

            self.long_term_memory_index = FAISS.from_documents(
                [memory_document], self.embeddings
            )
        else:
            self.long_term_memory_index.add_documents(
                [memory_document], encoder=self.embeddings
            )

    def query_long_term_memory(self, query, k=3) -> list:
        """Query the long-term memory for similar documents.  Returns a list of Document objects."""
        try:
            memories = self.long_term_memory_index.max_marginal_relevance_search(
                query, k=k
            )
            sorted_memories = sorted(memories, key=lambda x: x.metadata["id"])
            return sorted_memories
        except Exception as e:
            print("no memories found")
            print(e)
            return []

    def count_cost(self, result, model, summary=False) -> float:
        """Count the cost of the messages.
        The cost is calculated as the number of tokens in the input and output times the cost per token.
        Returns the cost."""

        # cost is calculated as the number of tokens in the input and output times the cost per token
        if model.startswith("gpt-3"):
            input_cost = 0.0005 / 1000
            output_cost = 0.0015 / 1000
        elif model.startswith("gpt-4o-mini"):
            input_cost = 0.00015 / 1000
            output_cost = 0.0006 / 1000
        elif model.startswith("gpt-4"):
            input_cost = 0.01 / 1000
            output_cost = 0.03 / 1000
        elif "7b" or "8b" in model.lower() and "8x7b" not in model:
            input_cost = 0.0002 / 1000
            output_cost = 0.0002 / 1000
        elif "openchat/openchat-3.5-1210" in model.lower():
            input_cost = 0.0002 / 1000
            output_cost = 0.0002 / 1000
        elif "llama-2-13b" in model.lower():
            input_cost = 0.000225 / 1000
            output_cost = 0.000225 / 1000
        elif "13b" in model.lower():
            input_cost = 0.0003 / 1000
            output_cost = 0.0003 / 1000
        elif "gemini" in model:
            input_cost = 0
            output_cost = 0
        elif "claude" in model:
            input_cost = 0.00025 / 1000
            output_cost = 0.00125 / 1000
        else:
            print("Model not recognized")
            input_cost = 0
            output_cost = 0

        if "gemini" in model:
            messages = self.format_messages_for_gemini(self.messages)
            agent = genai.GenerativeModel(model_name=model)
            input_tokens = agent.count_tokens(messages[:-1]).total_tokens
            output_tokens = agent.count_tokens([messages[-1]]).total_tokens

        elif "claude" in model:
            input_tokens = result.usage.input_tokens
            output_tokens = result.usage.output_tokens

        else:
            input_tokens = result.usage.prompt_tokens
            output_tokens = result.usage.completion_tokens

        total_tokens = input_tokens + output_tokens
        lastest_cost = input_cost * input_tokens + output_cost * output_tokens

        self.total_cost += lastest_cost
        # determine the length of inputs and outputs
        self.average_cost = self.total_cost / (len(self.chat_history) / 2)
        self.total_tokens += total_tokens
        self.average_tokens = self.total_tokens / (len(self.chat_history) / 2)
        if not summary:
            self.current_memory_tokens = total_tokens

        # calculate the cost
        return lastest_cost

    def format_messages_for_gemini(self, messages) -> list:
        """Format the messages for the Gemini model.  Returns a list of messages."""
        messages = [
            {"role": message["role"], "parts": message["content"]}
            for message in messages
        ]
        for message in messages:
            if message["role"] == "assistant":
                message["role"] = "model"
        return messages

    def query(
        self,
        prompt,
        temperature=0.3,
        top_p=None,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=200,
    ) -> str:
        """Query the model for a response to a prompt.  The prompt is a string of text that the AI will respond to.
        The temperature is the degree of randomness of the model's output.  The lower the temperature, the more deterministic the output.
        The higher the temperature, the more random the output.  The default temperature is .3.  The response is a string of text.
        """
        os.write(1, f"MODEL = {self.model}\n".encode())

        prompt = f"[{self.user_name}]: {prompt} "
        self.messages = []
        # build the full model prompt
        # Query the long-term memory for similar documents
        if hasattr(self, "long_term_memory_index"):
            returned_memories = self.query_long_term_memory(prompt)
            if len(returned_memories) > 0:
                # convert the memories to a string
                retrieved_memories = {
                    doc.page_content for doc in returned_memories
                }  # Remove duplicate memories
                self.long_term_memories = " : ".join(retrieved_memories)
            else:
                print("no memories retrieved")
        else:
            print("no memories yet")

        self.set_system_message()

        self.messages.append(self.system_message)
        self.messages.extend(self.short_term_memory)
        self.messages.append(
            {
                "role": "user",
                "content": self.start_ins + self.prefix + prompt + self.end_ins,
            }
        )
        # Query the model through the API
        if "gemini" in self.model:
            # attempt to query the Gemini model
            # configure the model with system instruction
            self.agent = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.messages[0]["content"],
            )

            # configuration
            config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            )
            if self.nsfw:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = None
            # format the messages for the Gemini model
            gemini_messages = self.format_messages_for_gemini(self.messages[1:])
            ## attempt to query the model
            query_successful = False
            tries = 0
            while not query_successful and tries <= 5:

                try:
                    result = self.agent.generate_content(
                        contents=gemini_messages,
                        generation_config=config,
                        safety_settings=safety_settings,
                    )
                    content = result.text
                    query_successful = True

                except:
                    tries += 1
                    print("Gemini model query failed, trying again")
                    print(f"response = {result}")
                    if len(result.candidates) > 0:
                        content = f"[Gemini]: I did not respond {result.candidates[0].finish_reason}.  Please adjust your prompt and try again"
                    else:
                        content = "[Gemini]: I did not respond.  Please adjust your prompt or change models and try again"
        elif "claude" in self.model:
            result = self.agent.messages.create(
                model=self.model,
                system=self.messages[0]["content"],
                messages=self.messages[1:],  # this is the conversation history
                temperature=temperature,  # this is the degree of randomness of the model's output
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = result.content[0].text
        else:
            result = self.agent.chat.completions.create(
                model=self.model,
                messages=self.messages,  # this is the conversation history
                temperature=temperature,  # this is the degree of randomness of the model's output
                frequency_penalty=frequency_penalty,  # This is the penalty for using a token based on frequency in the text.
                presence_penalty=presence_penalty,  # This is penalty for using a token based on its presence in the text.
                max_tokens=max_tokens,
                top_p=top_p,
            )
            content = result.choices[0].message.content

        # Check For NSFW Content
        if not self.nsfw:
            moderation = openai.OpenAI().moderations.create(input=content)
            flagged = moderation.results[0].flagged
            if flagged:

                return "[System]: I'm sorry, this response has been flagged as NSFW and cannot be shown."
            else:
                self.response = content
        else:
            # Store the response
            self.response = content

        if self.message_style_sample == None:
            self.message_style_sample = f"An example of how your character speaks is here inside triple backticks ```{self.response}```"

        # add response to current message history
        self.messages.append({"role": "assistant", "content": self.response})

        # Add user prompt to message history
        self.add_message(prompt, role="user")

        # Add reply to message history
        self.add_message(self.response, role="assistant")

        self.count_cost(result, self.model)

        if len(self.short_term_memory) >= self.max_short_term_memory_length:
            self.summarize_memories()

        return self.response

    def clear_history(self):
        """Clear the AI's memory.  Returns nothing."""
        self.short_term_memory = []
        self.chat_history = []
        self.mid_term_memory = "nothing yet."
        self.long_term_memories = "nothing yet."
        self.current_memory_id = 0
        self.message_style_sample = None
        self.prefix = ""
        self.response = "I'm thinking of my response"
        self.messages = []
        self.total_cost = 0
        self.total_tokens = 0
        self.current_memory_tokens = 0
        self.average_tokens = 0
        if hasattr(self, "long_term_memory_index"):
            del self.long_term_memory_index
        self.set_system_message()

    def get_memory(self):
        """Return the AI's current memory.  Returns a list of messages."""
        return self.messages

    def get_history(self):
        """Return the AI's full chat history.  Returns a list of messages."""
        return self.chat_history

    def save_agent(self):
        """Save agent to path"""
        saved_attrs = self.__dict__.copy()
        del saved_attrs["system_prompt"]
        del saved_attrs["summary_agent"]
        del saved_attrs["agent"]
        del saved_attrs["embeddings"]
        if "long_term_memory_index" in saved_attrs.keys():
            saved_attrs["long_term_memory_index"] = saved_attrs[
                "long_term_memory_index"
            ].serialize_to_bytes()
        return pickle.dumps(saved_attrs)

    def load_agent(self, file):
        """Load saved agent from path"""
        loaded_attrs = pickle.loads(file)

        # Replace agent attributes with loaded attribute's
        attrs_to_set = [
            "system_message",
            "character",
            "location",
            "user_name",
            "character_name",
            "short_term_memory",
            "chat_history",
            "mid_term_memory",
            "current_memory_id",
            "message_style_sample",
            "prefix",
            "messages",
            "total_cost",
            "total_tokens",
            "current_memory_tokens",
            "average_tokens",
        ]
        for attr in attrs_to_set:
            if attr in self.__dict__.keys():
                setattr(self, attr, loaded_attrs[attr])

        # for key, value in loaded_attrs.items():
        #     if key in self.__dict__.keys():
        #         setattr(self, key, value)

        self.set_summary_model(self.summary_model)
        self.set_model(self.model)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # De-serialize the vector db
        if "long_term_memory_index" in loaded_attrs:
            self.long_term_memory_index = FAISS.deserialize_from_bytes(
                loaded_attrs["long_term_memory_index"],
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
