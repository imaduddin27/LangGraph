from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import operator

load_dotenv()   

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])],
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)

app = graph.compile()
print("\nType exit/quit/bye to end chat. \n\nNote: The bot does not retain memory between interactions and does not have access to online tools.")
while True:
    user_input = input("You: ")
    if user_input in ["exit", "quit", "bye"]:
        print("Goodbye! See you next time.")
        break
    else:
        response = app.invoke({
            "messages": [
                HumanMessage(content=user_input),
            ]
        })
        #print(response) # Uncomment to see the full result
        ai_message = response["messages"][1].content  # Accessing the second message (AI response)
        print("Assistant:", ai_message)
