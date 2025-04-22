from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

sqlite_conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)


search_tool = TavilySearchResults()
tools = [search_tool]

#llm = ChatGroq(model_name="llama-3.3-70b-versatile")
llm = ChatGroq(model_name="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)
memory = SqliteSaver(sqlite_conn)

class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]



def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

def tools_router(state: BasicChatBot):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatBot)
graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router)

graph.add_edge("tool_node", "chatbot")

app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id": 1
    }}
print("\nType exit/quit/bye to end chat. \n\nNote: The bot will retain memory between interactions and has access to online tools.")
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