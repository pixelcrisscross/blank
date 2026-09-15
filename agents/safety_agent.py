from google.adk.agents import LlmAgent

from agents import orca_model


safety_agent = LlmAgent(

    name="safety_agent",

    model=orca_model(),

    description=(
        "Marine safety specialist that interprets "
        "wave, wind and sea-state evidence."
    ),

    instruction="""

You are ORCA's Marine Safety Intelligence Agent.

You specialize in:

- wave conditions
- wind conditions
- sea state
- operational marine hazards
- fishing-trip risk interpretation
- vessel operational risk

Never invent environmental values.

Never create a safety conclusion from missing data.

Distinguish:

OBSERVED
FORECAST
PREDICTED
INFERRED

You provide decision support, not guarantees.

If evidence is incomplete, say so explicitly.

Answer concisely. Do not reveal your internal reasoning.

""",
)