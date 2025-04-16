from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Get the current system date and time in the specified format.
               
    Examples:
        - For date only: get_system_time(format="%Y-%m-%d")
        - For date and time: get_system_time(format="%Y-%m-%d %H:%M:%S")
        - For time only: get_system_time(format="%H:%M:%S")
        
    Returns:
        The current date/time formatted according to the specified format.
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]

agent= initialize_agent(llm=llm, tools=tools, agent="zero-shot-react-description", verbose= True)

agent.invoke("When was the last visit of the president of the united states to germany and howmany days was it ago")


