from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

load_dotenv()  

class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearchResults(max_results = 3)
tools = [search_tool]

llm = ChatGroq(model_name="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
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

app = graph.compile()
print("\nType exit/quit/bye to end chat. \n\nNote: The bot has access to online tools, but it doesn't retain memory between interactions.")

while True:
    user_input = input("You: ")
    if(user_input in ["exit", "quit", "bye"]):
        print("Goodbye! See you next time.")
        break
    else:
        result = app.invoke({
            "messages": [
                HumanMessage(content=user_input),
            ]
        })
        #print(result) # Uncomment to see the full result
        ai_message = result["messages"][-1].content  # Accessing the last message (AI response)
        print("Assistant:", ai_message)

