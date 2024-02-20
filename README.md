# long-memory-character-chat
 A character chat with integrated medium and long-term memory

# Problem Statement
Character based chats are great and engrossing, but they have short memories.  They often lose track of the setting, situation, and important prior events.  While using longer context windows can help this, tokens are expensive!  How can we give a character a long memory, but limit the number of tokens sent for each completion?

# Solution
I hope to use a conversation summary integrated into the system prompt as a medium term memory to help the model remember recent events, and a vector database of stored long term memories to allow the model to recall events from a more distant past using vector similarity with current prompts.  My goal is to do this while limiting each message length sent to the model to between 1k and 2k tokens, which currently corresponds to $$0.0005 - $$0.0010 per message sent.

# Remaining Problems to Overcome:
1. How can the model keep track of the order of past events in long term memory?  ex. Did we raid the tower before or after killing the dragon?
   * timestamps included in summary?
   * message ids for sequence?
3. Should the medium term memory be 'rolling' in that the previous medium term memory summary should be included in the prompt to create the next one?  This would result in a decay of specificity of the medium term memory as the token limit for the summary is reached, but would also extend the historical length of the medium term memory.
   * pros: longer term medium term memory
   * cons: reduction in specificity of medium term memory
