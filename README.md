# long-memory-character-chat
 A character chat with integrated medium and long-term memory

 [Check it out here!](https://long-memory-character-chat.streamlit.app/)

# Problem Statements
1. Character based chats are great and engrossing, but they have short memories.  They often lose track of the setting, situation, and important prior events.  While using longer context windows can help this, tokens are expensive!  How can we give a character a long memory, but limit the number of tokens sent for each completion?

2. Characters lose their personality over time or tend to get stuck in repetitious response loops.

## Proble 1: Memory
### Solution
The Agent is provided with a multi-tiered "memory", or dynamically changing system prompt, which includes levels of information from previous conversation turns ranging from very specific for the most recent to more summarized for most distance topics and information.

**Short-term Memory Rolling Chat Window**
Each call to the model includes a fairly short list of previous messages verbatim.  This helps the model keep track of the context of the current conversation and recent messages.

**Mid-term Meory Scratchpad**
Further, I created a scratchpad for the AI agent to use to take notes to keep track of recent highlights in the conversation.  This scratchpad is updated on a rolling basis with new information added or updated and old, no longer relevant information discarded.  However, no information is actually lost because snapshots of his scratchpad are saved as the entries into the vector database.

Because this information is summarized, it compresses recent information, allowing the model to have the benefits of a longer chat window and larger context, while minimizing actual tokens included in the prompt.

Character names (not just 'user' and 'assistant') are included in each message to help with summarization.  This also helps the model keep track of who is saying what.  This makes the AI agent much less likely to get confused between who said what (which is otherwise an issue).  It also helps maintain turn-taking ettiquette in the dialogue and prevents the agent from speaking for the user (which can also be a problem)

**Long-Term Vector Store Memory**
I use conversation summaries to extract important information from conversations, and then store those in a vector database.  Each user query triggers a semantic search from the vector database to recall relevant information from previous conversation summaries.  This information is included in the prompt to the model, as well as a few of the most recent messages.

This way an AI agent's memory can far exceed its context.  In fact, as vector stores can take up relatively little disk space, the potential effective size of the agent's memory is functionally unlimited.

When memories are stored, they are stored with timestamps and when they are recalled they are ordered chronologically.  The agent is instructed to prefer more recent information if memories are conflicting.

**Result**
The model is able to accurately recall information from earlier in the conversation most of the time.  Vector store searches are generally successful in retrieving the relevant information, but the model does not always actually make use of the information in their responses.  Generally, though, I've been pleased with the results.

**Next Steps**
I'd like to get the model to more proactively reference previous events.  It is mostly successful in recall if I directly ask it questions about earlier conversations, but rarely spontaneously references previous events.  It lives very much in the moment.  I'm working on prompt engineering to generate this behavior, but it's still in the works.

## Personality/Conversation Style Degradation

I noticed that while the characters often started off strong with a unique speaking style, over time their style faded to be very bland and sound like their base AI.  I experimented with a few solutions, including having it write it's notice in character.  However, this wasted a lot of token in the memory by creating less token-dense notes.

Instead, I store a sample of the first response the model provides as an example style.  This first response is then passed to the agent on every prompt as an example speaking style for every prompt.  I have found this to be VERY effective in helping the model maintain speaking style consistently, even over long conversations (100+ completions).  

**Next Steps**
While the tone of the messages is consistent, I'd like to give the models even stronger, more nuanced and consistent personalities.  I suspect this may require fine-tuning.  
