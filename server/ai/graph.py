"""
LangGraph state graph for AI player decision-making.

4-step reasoning chain:
  1. Assess situation — "What's happening?"
  2. Review strategy — "Is my plan still valid?"
  3. Formulate tactics — "What should I do this trick?"
  4. Select cards — "Which cards do I play?"

The graph uses conditional edges to skip Step 3 when the strategy
is still valid, saving an API call.
"""

import json
import re
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END
from openai import OpenAI

from .prompts import (
    SYSTEM_PROMPT,
    assess_situation_prompt,
    review_strategy_prompt,
    formulate_strategy_prompt,
    select_cards_prompt,
    build_role_description,
)
from .session_manager import AISession


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph."""
    # Input
    player_index: int
    phase: str
    game_state: dict
    hand: list[dict]
    legal_actions: list[str]
    session: Any  # AISession
    model: str
    api_key: str
    base_url: str

    # Intermediate
    assessment: Optional[dict]
    strategy_review: Optional[dict]
    strategy_plan: Optional[dict]

    # Output
    action_type: str
    card_ids: list[str]
    reasoning: str


def build_ai_graph() -> StateGraph:
    """Build the LangGraph state graph for AI decision-making."""
    graph = StateGraph(AgentState)

    graph.add_node("assess", assess_node)
    graph.add_node("review", review_node)
    graph.add_node("formulate", formulate_node)
    graph.add_node("select", select_node)

    graph.set_entry_point("assess")
    graph.add_edge("assess", "review")

    # Condition: if strategy is still valid, skip to card selection
    graph.add_conditional_edges(
        "review",
        lambda state: "select" if state.get("strategy_review", {}).get("still_valid")
        else "formulate",
        {"select": "select", "formulate": "formulate"},
    )

    graph.add_edge("formulate", "select")
    graph.add_edge("select", END)

    return graph


def _get_client(state: AgentState) -> OpenAI:
    """Create OpenAI client from state."""
    return OpenAI(
        api_key=state.get("api_key", ""),
        base_url=state.get("base_url", "https://api.openai.com/v1"),
    )


def _call_llm(client: OpenAI, model: str, system: str, user: str,
              response_format: Optional[dict] = None) -> dict:
    """Call the LLM and return parsed JSON response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 500,
    }

    if response_format:
        kwargs["response_format"] = response_format

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"

        # Extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {}
    except Exception as e:
        print(f"LLM call error: {e}")
        return {}


def _get_role_description(state: AgentState) -> str:
    """Determine the role description for this player."""
    gs = state.get("game_state", {})
    player_index = state.get("player_index", 0)
    team_index = gs.get("my_team_index", 0)
    is_declarer = gs.get("my_is_declarer", False)
    return build_role_description(player_index, team_index, is_declarer)


# ---- Nodes ----

def assess_node(state: AgentState) -> AgentState:
    """Step 1: Assess the current situation."""
    gs = state.get("game_state", {})
    hand = state.get("hand", [])

    user_prompt = assess_situation_prompt(gs, hand)

    client = _get_client(state)
    role = _get_role_description(state)
    system = SYSTEM_PROMPT.format(role_description=role)

    # For assessment, use a simple text call
    result = _call_llm(client, state.get("model", "gpt-4o"), system, user_prompt)

    state["assessment"] = result if result else {"note": "assessment skipped"}
    return state


def review_node(state: AgentState) -> AgentState:
    """Step 2: Review if current strategy is still valid."""
    gs = state.get("game_state", {})
    hand = state.get("hand", [])
    session = state.get("session")

    previous = session.strategy if session else None
    memories = session.key_memories if session else []

    user_prompt = review_strategy_prompt(gs, hand, previous, memories)

    client = _get_client(state)
    role = _get_role_description(state)
    system = SYSTEM_PROMPT.format(role_description=role)

    result = _call_llm(client, state.get("model", "gpt-4o"), system, user_prompt)

    state["strategy_review"] = result if result else {"still_valid": True}

    # Update session
    if session and result and isinstance(result, dict):
        if not result.get("still_valid") and result.get("new_strategy"):
            session.update_strategy({"name": result["new_strategy"], "set_at_trick": gs.get("trick_count", 0)})

    return state


def formulate_node(state: AgentState) -> AgentState:
    """Step 3: Formulate specific tactics for this trick."""
    gs = state.get("game_state", {})
    hand = state.get("hand", [])
    session = state.get("session")

    review = state.get("strategy_review", {})
    strategy_name = review.get("new_strategy", review.get("previous_strategy", "default"))

    opponent_models = session.opponent_models if session else {}

    user_prompt = formulate_strategy_prompt(gs, hand, strategy_name, opponent_models)

    client = _get_client(state)
    role = _get_role_description(state)
    system = SYSTEM_PROMPT.format(role_description=role)

    result = _call_llm(client, state.get("model", "gpt-4o"), system, user_prompt)

    state["strategy_plan"] = result if result else {"strategy": strategy_name}

    # Update session with new strategy plan
    if session and result:
        session.update_strategy(result)

    return state


def select_node(state: AgentState) -> AgentState:
    """Step 4: Select specific cards to play."""
    gs = state.get("game_state", {})
    hand = state.get("hand", [])
    legal_actions = state.get("legal_actions", [])

    strategy_plan = state.get("strategy_plan") or state.get("strategy_review")

    user_prompt = select_cards_prompt(gs, hand, legal_actions, strategy_plan)

    client = _get_client(state)
    role = _get_role_description(state)
    system = SYSTEM_PROMPT.format(role_description=role)

    result = _call_llm(client, state.get("model", "gpt-4o"), system, user_prompt)

    card_ids = result.get("card_ids", []) if result else []
    reasoning = result.get("reasoning", "基于当前策略选择") if result else "Fallback selection"

    # If no cards selected, pick first legal action's first card
    if not card_ids and hand:
        card_ids = [hand[0].get("id", "")]

    state["action_type"] = "play"
    state["card_ids"] = card_ids
    state["reasoning"] = reasoning

    # Save session
    session = state.get("session")
    if session:
        from .session_manager import session_manager
        session_manager.save(state.get("player_index", 0))

    return state


# ---- Run the graph ----

_compiled_graph: Optional[StateGraph] = None


def get_graph() -> StateGraph:
    """Get or create the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_ai_graph().compile()
    return _compiled_graph


async def run_ai_decision(
    player_index: int,
    phase: str,
    game_state: dict,
    hand: list[dict],
    legal_actions: list[str],
    session: AISession,
    model: str = "gpt-4o",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
) -> dict:
    """Run the full AI decision pipeline."""
    initial_state: AgentState = {
        "player_index": player_index,
        "phase": phase,
        "game_state": game_state,
        "hand": hand,
        "legal_actions": legal_actions,
        "session": session,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "assessment": None,
        "strategy_review": None,
        "strategy_plan": None,
        "action_type": "play",
        "card_ids": [],
        "reasoning": "",
    }

    graph = get_graph()

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        print(f"LangGraph error: {e}, using fallback")
        result = {
            "action_type": "play",
            "card_ids": [hand[0].get("id")] if hand else [],
            "reasoning": f"Error: {e}, fallback to first card",
        }

    return {
        "action_type": result.get("action_type", "play"),
        "card_ids": result.get("card_ids", []),
        "reasoning": result.get("reasoning", ""),
    }
