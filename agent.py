import asyncio
import gradio as gr
import openai
import os

from mcp import ClientSession
from mcp.client.sse import sse_client

# Load OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)


# --- AI Tool Selector ---
def get_tool_from_ai(user_input: str) -> str:
    system_prompt = (
        "You are a smart tool router. Based on user input, return one of the tools:\n"
        "- summarize - summarize a big passage or essay into 2 to 3 brief sentences\n"
        "- greeting\n"
        "- ai_agent\n"
        "Only return the tool name. No explanation.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User: {user_input}\nTool:"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=5,
        temperature=0.2,
    )

    tool_name = response.choices[0].message.content.strip().lower()
    valid_tools = {"summarize", "search_knowledgebase", "greeting", "ai_agent"}
    return tool_name if tool_name in valid_tools else "ai_agent"


# --- Async MCP Tool Call ---
async def call_mcp_tool(user_input: str, selected_tool: str = None) -> tuple[str, str]:
    tool = selected_tool or get_tool_from_ai(user_input)
    print(f"Tool selected: {tool}")

    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            if tool == "greeting":
                name = user_input.split()[-1].capitalize()
                result = await session.read_resource(f"greeting://{name}")
                return tool, "".join([c.text for c in result.contents])

            elif tool == "search_knowledgebase":
                result = await session.call_tool(tool, arguments={"query": user_input})
                return tool, "".join([c.text for c in result.content])

            elif tool == "summarize":
                result = await session.call_tool(tool, arguments={"text": user_input})
                return tool, "".join([c.text for c in result.content])

            else:  # ai_agent fallback
                result = await session.call_tool("ai_agent", arguments={"question": user_input})
                return tool, "".join([c.text for c in result.content])


# --- Gradio UI ---
def launch_gradio():
    def ask(user_input):
        tool, output = asyncio.run(call_mcp_tool(user_input))
        return f"[Tool: {tool}]\n\n{output}"

    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 MCP Agent (AI Tool Routing Enabled)")
        input_box = gr.Textbox(label="Ask me anything", lines=2, placeholder="e.g. summarize this text...")
        output_box = gr.Textbox(label="Response", lines=10)
        ask_button = gr.Button("Submit")

        ask_button.click(fn=ask, inputs=input_box, outputs=output_box)

    # ✅ Use PORT and host 0.0.0.0 for Render
    port = int(os.environ.get("PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    launch_gradio()
