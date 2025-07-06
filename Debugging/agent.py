import os 
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
## Visualize the graph
from IPython.display import Image,display
load_dotenv()

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Langraph_Crash_Course"

llm = init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

## GRAPH with tool call

def make_tool_graph():
    @tool
    def add(a: int, b: int) -> int:
     """Add two numbers."""
     return a + b
    
    tools = [add]
    tool_node = ToolNode([add])
    llm_with_tool = llm.bind_tools([add])

    def call_llm_model(state: State):
        return {"messages": [llm_with_tool.invoke(state["messages"])]}
    

    builder = StateGraph(State)
    # Use a custom node name instead of START (reserved)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    ## Compile the graph
    graph = builder.compile()
    return graph

tool_agent = make_tool_graph() 

    

    

    
    


