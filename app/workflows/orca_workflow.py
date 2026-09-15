from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

from agents.planner import planner_agent
from agents.data_agents import (
    ResolveLocationAgent,
    CapabilityAgent,
    DynamicDataCollectionAgent,
    RiskAssessmentAgent,
    RecheckAgent,
    ReviewGateAgent,
    SimpleQueryGateAgent,
)
from agents.reasoning_agents import (
    ocean_reasoner,
    weather_reasoner,
    fishery_reasoner,
    safety_reasoner,
    tourism_reasoner,
    peer_review_agent,
)
from agents.review_agent import review_agent
from agents.synthesis_agent import synthesis_agent


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

specialist_parallel = ParallelAgent(
    name="parallel_specialist_reasoning",
    description="Run independent domain interpretations concurrently.",
    sub_agents=[
        ocean_reasoner,
        weather_reasoner,
        fishery_reasoner,
        safety_reasoner,
        tourism_reasoner,
    ],
)

risk_agent = RiskAssessmentAgent(
    name="deterministic_risk_assessment",
    description="Calculate a deterministic marine risk score.",
)

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


orca_workflow = SequentialAgent(
    name="orca_marine_workflow",
    description=(
        "Agentic marine-intelligence workflow with planning, dynamic "
        "evidence collection, specialist collaboration, iterative review, "
        "and synthesis."
    ),
    sub_agents=[
        planner_agent,
        capability_agent,
        resolve_location_agent,
        data_collection_agent,
        specialist_parallel,
        risk_agent,
        review_loop,
        synthesis_agent,
    ],
)