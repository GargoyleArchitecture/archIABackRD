
import re
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm
from src.graph.utils import _push_turn
from src.graph.consts import prompt_creator

def creator_node(state: GraphState) -> GraphState:
    user_q = state["userQuestion"]
    effective_q = state.get("localQuestion") or user_q

    prompt = f"""{prompt_creator}

User request:
{effective_q}

If an ASR is provided, ensure components and connectors explicitly support the Response and Response Measure.
"""
    _push_turn(state, role="system", name="creator_system", content=prompt)

    response = llm.invoke(prompt)
    content = getattr(response, "content", "")

    match = re.search(r"```mermaid\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    mermaid_code = (match.group(1).strip() if match else "").strip()

    _push_turn(state, role="assistant", name="creator", content=content)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content, name="creator")],
        "mermaidCode": mermaid_code,
        "hasVisitedCreator": True
    }
