from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflector_chain

load_dotenv()
REFLECT = "reflect"
GENERATE = "generate"

graph = MessageGraph()



def generate_node(state):
    return generation_chain.invoke({
        "messages": state
        })

def reflect_node(state):
    response = reflector_chain.invoke({
        "messages": state
        })
    return [HumanMessage(content=response.content)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def Should_cont(state):
    if(len(state) > 6):
        return END
    return REFLECT


graph.add_conditional_edges(GENERATE, Should_cont)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

print(response)







