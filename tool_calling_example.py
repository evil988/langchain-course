from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import Tool, tool
from langchain_groq import ChatGroq

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    print("Hello Tool Calling LangChain!")
    tools: List[Tool] = [get_text_length]

    # Create model and bind tools (native tool calling)
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    ).bind_tools(tools)

    # Start conversation
    messages = [
        HumanMessage(content="What is the length of the word: DOG"),
    ]

    # First response
    response: AIMessage = model.invoke(messages)
    print(response)

    # Execute tool calls iteratively
    max_rounds = 5
    rounds = 0
    while getattr(response, "tool_calls", None) and rounds < max_rounds:
        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            tool_map = {t.name: t for t in tools}
            if name not in tool_map:
                raise ValueError(f"Unknown tool requested: {name}")
            result = tool_map[name].invoke(args)
            messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", "call_0")))

        response = model.invoke(messages)
        print(response)
        rounds += 1

    # Final answer
    print(getattr(response, "content", ""))
