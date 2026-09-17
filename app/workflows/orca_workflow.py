from google.adk.agents import SequentialAgent, LoopAgent

from agents.language_agent import LanguageDetectionAgent
from agents.planner import planner_agent
from agents.data_agents import (
    ResolveLocationAgent,
    CapabilityAgent,
    DynamicDataCollectionAgent,
    ConditionalReasoningAgent,
    RiskAssessmentAgent,
    RecheckAgent,
    ReviewGateAgent,
    SimpleQueryGateAgent,
)
from agents.reasoning_agents import peer_review_agent
from agents.review_agent import review_agent
from agents.route_agent import RouteSafetyAgent
from agents.synthesis_agent import synthesis_agent


# ── Phase 2: Language Detection ──────────────────────────────────────
language_detection_agent = LanguageDetectionAgent(
    name="language_detection",
    description=(
        "Detects the user's language (including Indian regional languages) "
        "and stores it in session state so the synthesizer can respond in "
        "the same language."
    ),
)

# ── Core pipeline agents ─────────────────────────────────────────────
capability_agent = CapabilityAgent(
    name="capability_agent",
    description="Short-circuits meta-queries with a capability list.",
)

resolve_location_agent = ResolveLocationAgent(
    name="resolve_location",
    description="Resolve the location before data retrieval.",
)

data_collection_agent = DynamicDataCollectionAgent(
    name="dynamic_data_collection",
    description=(
        "Run only the data collectors the planner requested, in parallel. "
        "Skips marine domains when the location is inland."
    ),
)

reasoning_agent = ConditionalReasoningAgent(
    name="conditional_reasoning",
    description=(
        "Run only the specialist reasoners whose domains were collected."
    ),
)

risk_agent = RiskAssessmentAgent(
    name="deterministic_risk_assessment",
    description="Calculate a deterministic marine risk score.",
)

# ── Phase 3: Route Safety ─────────────────────────────────────────────
route_safety_agent = RouteSafetyAgent(
    name="route_safety_assessment",
    description=(
        "Evaluate marine route safety waypoint by waypoint. "
        "Only activates when planner intent is 'route'."
    ),
)

# ── Review loop ───────────────────────────────────────────────────────
review_loop = LoopAgent(
    name="evidence_review_loop",
    description="Iteratively review evidence.",
    max_iterations=2,
    sub_agents=[
        SimpleQueryGateAgent(
            name="simple_query_gate",
            description="Short-circuits review for single-domain queries.",
        ),
        peer_review_agent,
        review_agent,
        RecheckAgent(
            name="targeted_recheck",
            description="Re-fetch only domains identified by the quality reviewer.",
        ),
        ReviewGateAgent(
            name="review_exit_gate",
            description="Stops the review loop when evidence is sufficient or blocked.",
        ),
    ],
)


# ── Full ORCA workflow ────────────────────────────────────────────────
orca_workflow = SequentialAgent(
    name="orca_marine_workflow",
    description=(
        "Agentic marine-intelligence workflow with language detection, "
        "planning, dynamic evidence collection, conditional specialist "
        "reasoning, route safety analysis, iterative review, and "
        "multilingual synthesis."
    ),
    sub_agents=[
        language_detection_agent,   # Phase 2: detect user language first
        planner_agent,
        capability_agent,
        resolve_location_agent,
        data_collection_agent,
        reasoning_agent,
        risk_agent,
        route_safety_agent,         # Phase 3: route safety (skips if not route intent)
        review_loop,
        synthesis_agent,
    ],
)