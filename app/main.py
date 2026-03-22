import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas.models import QuestionRequest, AgentResponse
from app.agent.graph import agent
from app.agent.state import AgentState

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Service liveness check."""
    return {"status": "ok", "service": "country-agent"}


@app.post("/ask", response_model=AgentResponse, tags=["Agent"])
async def ask_agent(request: QuestionRequest):
    """
    Submit a natural language question about a country.

    The agent will:
    1. Parse your intent and identify the country + fields
    2. Fetch live data from the REST Countries API
    3. Synthesize a grounded, accurate answer
    """
    logger.info(f"Incoming question: '{request.question}'")

    initial_state: AgentState = {
        "user_question": request.question,
        "country_name": None,
        "requested_fields": None,
        "raw_country_data": None,
        "final_answer": None,
        "error": None,
    }

    try:
        result: AgentState = await agent.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Agent pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an unexpected error. Please try again.",
        )

    return AgentResponse(
        question=request.question,
        answer=result.get("final_answer", "No answer was generated."),
        country_detected=result.get("country_name"),
        fields_requested=result.get("requested_fields"),
    )