from typing import List
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from chains import parser
from langgraph.prebuilt import ToolInvocation

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    last_ai_message: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(last_ai_message)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json", 
                    tool_input=query,
                )
            )
            ids.append(parsed_call["id"])

    



raw_res = execute_tools(
        state=[
            HumanMessage(
                content="Write about how small business can leverage AI to grow"
            ),
            AIMessage(
                content="", 
                tool_calls=[
                    {
                        "name": "AnswerQuestion",
                        "args": {
                            'answer': '', 
                            'search_queries': [
                                    'AI tools for small business', 
                                    'AI in small business marketing', 
                                    'AI automation for small business'
                            ], 
                            'reflection': {
                                'missing': '', 
                                'superfluous': ''
                            }
                        },
                        "id": "call_KpYHichFFEmLitHFvFhKy1Ra",
                    }
                ],
            ),
        ]
    )