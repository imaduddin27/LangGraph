from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from schema import AnswerQuestion, RevisedAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
parser = JsonOutputToolsParser(return_id=True)

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
#llm = ChatGroq(model_name="llama-3.2-90b-vision-preview")


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are expert AI researcher.
                Current time: {time}
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. After the reflection,  **list 1-3 search queries saperately** for researching improvements. Do not include them inside reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice = 'AnswerQuestion') | pydantic_parser

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)

first_responder_chain = revisor_prompt_template | llm.bind_tools(tools=[RevisedAnswer], tool_choice = 'RevisedAnswer')


response = first_responder_chain.invoke({
    "messages": [HumanMessage(content = "Write me a blog post on how small business can use AI to grow")]
})

print(response)