from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
memory = MemorySaver()
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])],
    }

graph = StateGraph(BasicChatState)  
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id": 1
    }}
print("\nType exit/quit/bye to end chat. \n\nNote: The bot will retain memory between interactions.")
while True:
    user_input = input("You: ")
    if user_input in ["exit","bye","end"]:
        print("Goodbye! See you next time.")
        break
    else:
        response = app.invoke({
            "messages": [HumanMessage(content=user_input)],
        },config=config)

        print("Assistant:", response["messages"][-1].content)  # Accessing the last message (AI response)