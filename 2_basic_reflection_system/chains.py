from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
#llm = ChatGroq(model_name="llama-3.2-90b-vision-preview")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a helpful assistant that can answer questions and help with tasks."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revosed version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tet. Generate critique and recommendations for each and every attempt of the users tweet."
            "Always provide detailed recommendations, including requests for lengh, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


generation_chain = generation_prompt | llm 
reflector_chain = reflector_prompt | llm 


