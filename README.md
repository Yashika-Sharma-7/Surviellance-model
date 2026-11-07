# Park Surveillance & Health Assessment

A local image-analysis application for a park-management demonstration. It
combines an existing YOLO segmentation model, an existing YOLO object detector,
and an optional OpenRouter language-model narrative. Python calculates the
health scores; the language model does not set or alter them.

## System architecture

1. `main.py` validates a single uploaded park or drone image and presents the
   Streamlit interface.
2. `Coverage_Calculation.py` loads `models/best.pt`, segments the image, assigns
   overlapping pixels to the highest-confidence mask, and returns a Markdown
   report, annotated image, and numeric coverage percentages.
3. `Count_Calculation.py` loads `models/Count.pt` and returns an object-count
   report, annotated image, and counts using the model's original class names.
4. `Park_Health.py` calculates provisional numerical scores in Python and
   optionally sends the measured coverage and counts to OpenRouter for written
   interpretations and recommendations.

The analysis uses one image; it is not a park-wide survey. Training scripts
remain separate and are not run by the application.

## Models and classes

The application uses the existing trained weights without retraining or
modifying them:

- **Segmentation** — `models/best.pt`
  1. Dead Vegetation
  2. Dry Area
  3. Flower bed
  4. Lawn
  5. Pathway
  6. Vegetation
  7. Water Body
  8. Weeds
  9. litter
- **Object detection** — `models/Count.pt`
  1. Bench
  2. Dustbin
  3. Light_pole
  4. Tree

## Installation

From the project directory, create and activate a virtual environment, then
install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build compatible with the local CUDA
setup using the official PyTorch installation selector. CPU inference is
available when CUDA is not configured.

## OpenRouter configuration

The narrative assessment uses the existing
`nvidia/nemotron-3.5-super-vl:free` model through OpenRouter. Create your local
`.env` from the safe template, then add your key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `OPENROUTER_API_KEY` in `.env`. Never commit `.env` or paste an API key into
source code. `.gitignore` excludes `.env` and other `.env.*` files while
allowing the placeholder `.env.example` to be committed. If the key is absent
or the OpenRouter request fails, the application still displays locally
calculated scores and reports that the generated narrative is unavailable.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Upload a JPEG, PNG, BMP, or WebP image up to 20 MiB (40 megapixels), then select
**Analyze image**.

## Expected outputs

- Original uploaded image.
- Segmentation-annotated and object-detection-annotated images.
- Numeric percentage coverage for lawn, tree canopy/vegetation, dry area,
  weeds, litter, dead vegetation, flower beds, water bodies, and pathways.
- Counts for Bench, Dustbin, Light_pole, and Tree.
- Overall park health and lawn quality scores on a 0–100 scale.
- Overall, lawn, vegetation, cleanliness, and infrastructure assessments,
  key observations, maintenance recommendations, and limitations.
- Markdown reports from both computer-vision modules.

## Scoring and limitations

The scores are **provisional heuristics** and have not been validated against
expert park assessments. They start at 50, apply percentage-point weights, are
clamped to 0–100, and rounded to two decimals:

- Overall park health = `50 + 0.6*lawn + 0.6*vegetation + 0.3*flower_bed
  - 1.2*weeds - dry_area - 1.2*dead_vegetation - 2.0*litter`.
- Lawn quality = `50 + 1.2*lawn + 0.3*flower_bed - 1.5*weeds - dry_area
  - 1.2*dead_vegetation`.

Water-body and pathway coverage and object counts are reported but are not
scored. They lack contextual inputs needed for a defensible adequacy score.
Tree counts do not measure tree health. Infrastructure counts do not establish
adequacy without park area, visitor numbers, and placement information. The
models may miss or misclassify content due to occlusion, image quality, and
viewpoint. No disease detection is performed. Confirm important findings with
on-site inspection and expert assessment.
