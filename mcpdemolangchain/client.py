from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is not None:
        os.environ["GROQ_API_KEY"] = groq_api_key
else:
        raise ValueError("GROQ_API_KEY is not set")


async def main():
    client = MultiServerMCPClient({ 
        "math": {
         "command": "python",
        "args": ["mathserver.py"],
        "transport": "stdio",
    },
    "weather": {
        "command": "python",
        "args": ["http://localhost:8000/mcp"],
        "transport": "streamable-http",
    }
})
    tools = await client.get_tools()
    model = ChatGroq(model = "qwen-qwq-32b")

    agent = create_react_agent(model, tools)

    math_response = await agent.ainvoke({"messages":[{"role":"user", "content":"What is 10 + 20?"}]})

    print("Math Response: ",math_response['messages'][-1].content)

asyncio.run(main())
