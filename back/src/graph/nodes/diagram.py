
import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.state import GraphState
from src.graph.resources import llm, log
from src.graph.consts import MERMAID_SYSTEM
from src.graph.utils import _sanitize_mermaid

def _llm_nl_to_mermaid(natural_prompt: str) -> str:
    """
    Llama al LLM para obtener código Mermaid puro (sin fences) y lo sanea
    con _sanitize_mermaid antes de devolverlo.
    """
    msgs = [SystemMessage(content=MERMAID_SYSTEM),
            HumanMessage(content=natural_prompt)]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp)) or ""

    # Si vino con ```mermaid ...```, usamos solo el cuerpo
    m = re.search(r"```mermaid\s*(.*?)```", raw, flags=re.I | re.S)
    if m:
        return _sanitize_mermaid(m.group(1))

    # Si vino con ```algo ...```, también usamos solo el cuerpo
    m = re.search(r"```(?:\w+)?\s*(.*?)```", raw, flags=re.I | re.S)
    if m:
        return _sanitize_mermaid(m.group(1))

    # Si no hay fences, saneamos todo el texto
    return _sanitize_mermaid(raw)

def diagram_orchestrator_node(state: GraphState) -> GraphState:
    """
    Nodo orquestador de diagramas:
    - Usa el ASR + estilo + tácticas + contexto + memoria del grafo
    - Genera SOLO el script Mermaid (state["mermaidCode"])
    - NO llama a Kroki, ni a /diagram/nl, ni genera SVG/PNG
    """
    # Pregunta actual del usuario (si existe)
    user_q = (state.get("localQuestion") or state.get("userQuestion") or "").strip()

    # --- ASR ---
    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or ""
    ).strip()

    # --- Estilo arquitectónico ---
    style_text = (
        state.get("style")
        or state.get("selected_style")
        or state.get("last_style")
        or ""
    ).strip()

    # --- Tácticas: preferimos la estructura JSON; si no, el markdown ---
    tactics_names: list[str] = []

    tactics_struct = state.get("tactics_struct") or []
    if isinstance(tactics_struct, list):
        for it in tactics_struct:
            if isinstance(it, dict) and it.get("name"):
                tactics_names.append(str(it["name"]))

    if not tactics_names:
        tactics_md = (state.get("tactics_md") or "").strip()
        if tactics_md:
            for line in tactics_md.splitlines():
                line = re.sub(r"^\s*[-*]\s*", "", line).strip()
                if line:
                    tactics_names.append(line)

    # Limitar un poco para que el diagrama no explote
    tactics_names = tactics_names[:8]

    tactics_block = (
        "\n".join(f"- {t}" for t in tactics_names)
        if tactics_names
        else "- (no explicit tactics selected yet)"
    )

    # --- Contexto / memoria adicional ---
    add_context = (state.get("add_context") or "").strip()
    doc_context = (state.get("doc_context") or "").strip()
    memory_text = (state.get("memory_text") or "").strip()

    sections: list[str] = []

    if add_context:
        sections.append(f"Business / project context:\n{add_context}")

    if doc_context:
        sections.append(f"Project documents context (RAG):\n{doc_context}")

    if memory_text:
        sections.append(f"Conversation memory (ASR/style/tactics decisions):\n{memory_text}")

    sections.append(
        "Quality Attribute Scenario (ASR):\n"
        f"{asr_text or '(not explicitly defined; infer it from the context and user request)'}"
    )

    sections.append(
        "Chosen architecture style:\n"
        f"{style_text or '(not explicitly chosen; infer a reasonable style for the ASR).'}"
    )

    sections.append("Selected tactics:\n" + tactics_block)

    sections.append(
        "User diagram request:\n"
        + (user_q or "Generate a deployment/component diagram aligned with the ASR and tactics.")
    )

    full_prompt = "\n\n---\n\n".join(sections)

    # --- Llamar al LLM especializado en Mermaid ---
    try:
        mermaid_code = _llm_nl_to_mermaid(full_prompt)
    except Exception as e:
        log.warning("diagram_orchestrator_node: Mermaid generation failed: %s", e)
        mermaid_code = ""

    state["mermaidCode"] = mermaid_code or ""
    # Ya no usamos imágenes ni backend de figuras
    state["diagram"] = {}
    state["hasVisitedDiagram"] = True
    # Opcional: marcar intención de diagrama por si algo más lo usa
    state["intent"] = "diagram"

    return state
