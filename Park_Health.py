import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

MODEL_NAME = "nvidia/nemotron-3.5-super-vl:free"
SCORING_METHODOLOGY = (
    "Provisional heuristic, not validated against expert park assessments. "
    "Scores start at 50, apply the documented percentage-point weights, "
    "are clamped to 0-100, and rounded to two decimals. The overall score "
    "weights lawn +0.6, vegetation +0.6, flower bed +0.3, weeds -1.2, "
    "dry area -1.0, dead vegetation -1.2, and litter -2.0. The lawn score "
    "weights lawn +1.2, flower bed +0.3, weeds -1.5, dry area -1.0, and "
    "dead vegetation -1.2. Water-body and pathway coverage, object counts, "
    "and tree counts are reported but not included in these scores."
)

TEXT_FIELDS = (
    "overall_assessment",
    "lawn_assessment",
    "vegetation_assessment",
    "cleanliness_assessment",
    "infrastructure_assessment",
)
LIST_FIELDS = (
    "key_observations",
    "maintenance_recommendations",
    "limitations",
)
OBJECT_CLASSES = ("Bench", "Dustbin", "Light_pole", "Tree")


def _metric_value(coverage, name):
    """Return a finite coverage percentage, treating omitted metrics as zero."""
    try:
        value = float(coverage.get(name, 0))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Coverage metric {name!r} must be numeric.") from error

    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(
            f"Coverage metric {name!r} must be between 0 and 100."
        )
    return value


def clamp(value):
    """Keep heuristic scores between 0 and 100, rounded to two decimals."""
    return round(max(0, min(100, value)), 2)


def calculate_scores(coverage):
    """Calculate reproducible provisional heuristic scores from coverage percentages."""
    lawn = _metric_value(coverage, "Lawn Coverage")
    vegetation = _metric_value(
        coverage, "Tree Canopy Cover (Vegetation)"
    )
    flower = _metric_value(coverage, "Flower Bed")
    weeds = _metric_value(coverage, "Weed Infestation")
    dry = _metric_value(coverage, "Bare Patches (Dry Area)")
    dead = _metric_value(coverage, "Dead Vegetation")
    litter = _metric_value(coverage, "Litter Detection")

    lawn_score = clamp(
        50
        + (1.2 * lawn)
        + (0.3 * flower)
        - (1.5 * weeds)
        - (1.0 * dry)
        - (1.2 * dead)
    )

    park_score = clamp(
        50
        + (0.6 * lawn)
        + (0.6 * vegetation)
        + (0.3 * flower)
        - (1.2 * weeds)
        - (1.0 * dry)
        - (1.2 * dead)
        - (2.0 * litter)
    )

    return {
        "overall_park_health": park_score,
        "lawn_quality": lawn_score,
    }


def _fallback_assessment(scores, error_message):
    limitations = [
        "Scores use a provisional heuristic and have not been validated "
        "against expert park assessments.",
        "Tree detection counts do not measure tree health.",
        "Bench, dustbin, and light-pole counts alone do not establish "
        "infrastructure adequacy without park area, visitor numbers, "
        "and placement information.",
        "The models analyze one image and may miss objects or misclassify "
        "covered, distant, or visually ambiguous areas.",
        "No diagnostic evidence is available for disease assessment.",
    ]
    return {
        **scores,
        "scoring_methodology": SCORING_METHODOLOGY,
        "overall_assessment": (
            "The numerical scores are available, but an AI-generated "
            "narrative assessment could not be completed."
        ),
        "lawn_assessment": "AI-generated lawn assessment unavailable.",
        "vegetation_assessment": "AI-generated vegetation assessment unavailable.",
        "cleanliness_assessment": "AI-generated cleanliness assessment unavailable.",
        "infrastructure_assessment": (
            "AI-generated infrastructure assessment unavailable."
        ),
        "key_observations": [],
        "maintenance_recommendations": [],
        "limitations": limitations,
        "error": error_message,
    }


def _validate_llm_sections(payload):
    if not isinstance(payload, dict):
        raise ValueError("The park assessment response must be a JSON object.")

    sections = {}
    for field in TEXT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"The park assessment response is missing text field {field!r}."
            )
        sections[field] = value.strip()

    for field in LIST_FIELDS:
        value = payload.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(
                f"The park assessment response field {field!r} "
                "must be a list of strings."
            )
        sections[field] = value
    return sections


def generate_park_assessment(coverage, object_counts):
    """Return Python-scored results with LLM-generated explanatory sections."""
    scores = calculate_scores(coverage)
    if not isinstance(object_counts, dict) or set(object_counts) != set(
        OBJECT_CLASSES
    ):
        raise ValueError(
            "Object counts must contain exactly these model classes: "
            + ", ".join(OBJECT_CLASSES)
        )
    for class_name, count in object_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Object count for {class_name!r} must be a non-negative integer."
            )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return _fallback_assessment(
            scores,
            "OPENROUTER_API_KEY was not found in the environment or project .env.",
        )

    park_data = {
        "coverage_percentages": coverage,
        "object_counts": object_counts,
    }

    system_prompt = """
You are an evidence-based assistant interpreting outputs from park image
segmentation and object-detection models. Treat all supplied values as model
measurements, not ground truth. Generate concise explanations and practical,
appropriately cautious recommendations based only on the supplied data.

Do not calculate, repeat, or modify numerical scores. Python calculates the
scores separately. Do not infer tree health from tree counts. Counts of
benches, dustbins, light poles, or trees do not establish infrastructure
adequacy without park area, visitor numbers, and placement information.
Do not claim disease detection. Do not claim conditions that cannot be
observed in the supplied metrics. State when evidence is insufficient.

Return ONLY valid JSON with these fields:
{
  "overall_assessment": "",
  "lawn_assessment": "",
  "vegetation_assessment": "",
  "cleanliness_assessment": "",
  "infrastructure_assessment": "",
  "key_observations": [],
  "maintenance_recommendations": [],
  "limitations": []
}
"""
    user_prompt = (
        "Interpret the following measured model outputs. Coverage values are "
        "percentages of the image area assigned to each class. Object counts "
        "are detections in this image, not measurements of park-wide provision.\n\n"
        f"{json.dumps(park_data, indent=2)}"
    )

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "OpenRouter returned a missing or non-text park assessment."
            )
        sections = _validate_llm_sections(json.loads(content))
    except (OpenAIError, json.JSONDecodeError, ValueError, IndexError) as error:
        return _fallback_assessment(scores, str(error))

    return {
        **scores,
        **sections,
        "scoring_methodology": SCORING_METHODOLOGY,
    }
