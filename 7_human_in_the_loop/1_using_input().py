from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    

llm = ChatGroq(model="llama-3.3-70b-versatile")

GENERATE_POST = "generate_post"
POST = "post"
GET_REVIEW_DECISION = "get_review_decision"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State):
    return{
        "messages": [llm.invoke(state["messages"])],
    }

def get_review_decision(state: State):
    post_content = state["messages"][-1].content

    print("The current LinkedIN post content is: \n", post_content, "\n")

    decision = input("Do you want to post it? (yes/no):")
    if decision.lower() == "yes":
        return POST
    else: 
        return COLLECT_FEEDBACK
    
def post(state: State):
    final_content = state["messages"][-1].content

    print("Posting the following content to LinkedIn: \n", final_content, "\n")

def review_post(state: State):
    feedback = input("How can I improove this post?")
    return {"messages": [HumanMessage(content=feedback)]}

graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.set_entry_point(GENERATE_POST)
graph.add_node(POST, post)
graph.add_node(COLLECT_FEEDBACK, review_post)

graph.add_conditional_edges(GENERATE_POST, get_review_decision)

graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

result = app.invoke(
    {"messages": 
     [HumanMessage(content="Generate a LinkedIn post about the importance of AI in the workplace.")]
     }
)

print(result)