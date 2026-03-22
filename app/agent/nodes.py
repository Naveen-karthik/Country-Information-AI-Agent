import json
import logging
from app.agent.state import AgentState
from app.agent.tools import fetch_country_data
from app.core.llm import call_mistral

logger = logging.getLogger(__name__)


# ── Node 1: Intent Parser ──────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent parser for a Country Information Agent.

Extract the country name and the fields the user is asking about from their question.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

Format:
{
  "country_name": "<country name in English>",
  "requested_fields": ["<field1>", "<field2>"]
}

Supported fields:
  population, capital, currency, languages, region, subregion,
  flag, area, timezones, borders, calling_code

Rules:
- If the user asks for "everything" or is vague, include ALL supported fields.
- If no country is mentioned, set country_name to null.
- Normalize country names to standard English (e.g. "Deutschland" → "Germany").
- Return ONLY the JSON object. No preamble. No explanation."""


async def intent_parser_node(state: AgentState) -> AgentState:
    """
    Node 1 — Extracts country_name and requested_fields from the user's question.
    Uses Mistral to parse natural language intent into structured JSON.
    """
    logger.info(f"[intent_parser] question='{state['user_question']}'")

    try:
        raw_response = await call_mistral(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=state["user_question"],
        )

        # Strip markdown fences if model wraps in ```json ... ```
        cleaned = raw_response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)

        country_name = parsed.get("country_name")
        requested_fields = parsed.get("requested_fields", [])

        logger.info(f"[intent_parser] country='{country_name}', fields={requested_fields}")

        return {
            **state,
            "country_name": country_name,
            "requested_fields": requested_fields,
        }

    except json.JSONDecodeError as e:
        logger.error(f"[intent_parser] JSON parse failed: {e}")
        return {
            **state,
            "country_name": None,
            "requested_fields": [],
            "error": "I couldn't understand your question. Please mention a country and what you'd like to know.",
        }
    except RuntimeError as e:
        logger.error(f"[intent_parser] LLM error: {e}")
        return {
            **state,
            "country_name": None,
            "requested_fields": [],
            "error": str(e),
        }


# ── Node 2: Tool Invocation ────────────────────────────────────────────────────

async def tool_invocation_node(state: AgentState) -> AgentState:
    """
    Node 2 — Calls the REST Countries API with the extracted country name.
    Populates raw_country_data or sets an error.
    """
    if state.get("error"):
        logger.info("[tool_invocation] Skipping — upstream error exists.")
        return state

    country_name = state.get("country_name")

    if not country_name:
        return {
            **state,
            "error": "No country was identified in your question. Please mention a specific country.",
        }

    logger.info(f"[tool_invocation] Fetching data for: '{country_name}'")

    try:
        data = await fetch_country_data(country_name)

        if data is None:
            return {
                **state,
                "error": (
                    f"I couldn't find any information for '{country_name}'. "
                    "Please check the spelling or try the country's name in English."
                ),
            }

        logger.info(f"[tool_invocation] Data fetched successfully for '{country_name}'")
        return {**state, "raw_country_data": data}

    except RuntimeError as e:
        logger.error(f"[tool_invocation] API error: {e}")
        return {**state, "error": str(e)}


# ── Node 3: Answer Synthesizer ─────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """You are a helpful Country Information Assistant.

You will receive:
1. The user's original question
2. The fields they asked about
3. Raw JSON data about a country from a public API

Your task:
- Extract ONLY the requested fields from the data
- Present the information in a clear, friendly, conversational tone
- If a field is missing in the data, say it is not available
- Do NOT add any information that is not present in the data
- Do NOT hallucinate facts
- Keep the answer concise and well-structured"""


async def answer_synthesis_node(state: AgentState) -> AgentState:
    """
    Node 3 — Composes the final answer.
    On error, returns a clean user-facing message.
    On success, uses Mistral to synthesize from raw data.
    """
    if state.get("error"):
        logger.info(f"[answer_synthesis] Returning error message: {state['error']}")
        return {
            **state,
            "final_answer": f"⚠️ {state['error']}",
        }

    country_data = state["raw_country_data"]
    fields = state.get("requested_fields", [])

    user_prompt = f"""User question: {state['user_question']}

Requested fields: {fields}

Country data (raw JSON):
{json.dumps(country_data, indent=2)}

Please answer the user's question using only the data above."""

    logger.info(f"[answer_synthesis] Synthesizing answer for '{state.get('country_name')}'")

    try:
        answer = await call_mistral(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        return {**state, "final_answer": answer.strip()}

    except RuntimeError as e:
        logger.error(f"[answer_synthesis] LLM error: {e}")
        return {
            **state,
            "final_answer": "I retrieved the data but couldn't generate a response. Please try again.",
        }