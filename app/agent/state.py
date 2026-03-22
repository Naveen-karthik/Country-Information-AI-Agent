from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    user_question: str
    country_name: Optional[str]
    requested_fields: Optional[List[str]]
    raw_country_data: Optional[dict]
    final_answer: Optional[str]
    error: Optional[str]