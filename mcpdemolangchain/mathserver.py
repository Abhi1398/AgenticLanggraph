from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()  # type: ignore
def add(a: int, b: int)->int:
    """_summary_
     Add two numbers together
    """
    return a + b

@mcp.tool()  # type: ignore
def subtract(a: int, b: int) -> int:
    """_summary_
     Subtract two numbers together
    """
    return a - b

@mcp.tool()  # type: ignore
def multiply(a: int, b: int) -> int:
    """_summary_
     Multiply two numbers together
    """
    return a * b

@mcp.tool()  # type: ignore
def divide(a: int, b: int) -> int:
    """_summary_
     Divide two numbers together
    """
    return a // b

## transport='stdio' is used  to run the server and input/output is done locally
if __name__ == "__main__":
    mcp.run(transport="stdio")