from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import uuid

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

def model(state: State):
    """
    Here we are using the LLM to generate a LinkedIn post with human incorporation 
    """

    print("[model] Generating content...")
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]

    prompt = f"""
    LnkedIn Topic: {linkedin_topic}
    Human Feedback: {feedback[-1] if feedback else "No Feedback yet"}

    generate a structured and well-written LinkedIn post based on the given topic.
    
    Comnsider previous human feedback to refine the response.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content creator."),
        HumanMessage(content=prompt),
    ])

    generated_linkedin_post = response.content

    print("[model_node] Generated post:\n", generated_linkedin_post)

    return {
        "generated_post": [AIMessage(content=generated_linkedin_post)],
        "human_feedback": feedback,
    }

def human_node(state: State):
    """ 
    Human Intervention mode which loops back to model unless input is done"""

    print("\n [human_node] awaiting human feedback...")

    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post": generated_post,
            "message": "provide feedback or type 'done' to finish",
        }
    )
    print(f"[human_node] Received user feedback:", {user_feedback})

    if user_feedback.lower() == "done":
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalised"]}, goto="end_node")
    
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

def end_node(state: State):
    """End node"""
    print("\n[End Node] Process completed.")
    print("\n Final LinkedIn Post:", state["generated_post"][-1])
    print("Final Human Feedback:", state["human_feedback"])
    return {"generated_post": state["generated_post"], "human_feedback": state["human_feedback"]}

graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)    

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

checkpointer= MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {
    "thread_id": uuid.uuid4(), 
}}
    
linkedin_topic = input("Enter LinkedIn topic: ")
initial_state = {
    "linkedin_topic": linkedin_topic,
    "generated_post": [],
    "human_feedback": []
}


for post in app.stream(initial_state, config=thread_config):
    for node_id, value in post.items():
        if(node_id == "__interrupt__"):
            while True:

                user_feedback = input("Provide feedback or type 'done' to finish: ")
                
                app.invoke(Command(resume=user_feedback), config=thread_config)

                if user_feedback.lower() == "done":
                    break