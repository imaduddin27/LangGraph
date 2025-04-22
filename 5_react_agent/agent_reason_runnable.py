from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import tool, create_react_agent
from langchain_community.tools import TavilySearchResults
import datetime
from langchain import hub
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

search_tool = TavilySearchResults(search_depth="basic")
react_prompt = hub.pull("hwchase17/react")


tools = [get_system_time, search_tool]

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)