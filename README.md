# Multi-Modal Evidence Review System

AI-powered damage claim verification using vision-language models.

## Architecture
```
Claim Parser → Image Analyzer → Evidence Matcher → Risk Detector → Decision Engine → Explanation Generator

The system follows a modular evidence-review pipeline that processes claim conversations, image evidence, user history, and evidence requirements before generating the final output.csv.

Images remain the primary source of truth, while user history contributes only risk signals and never overrides clear visual evidence.
```

## Key Design Decisions

- **Images are truth**: The Evidence Matcher always prioritises visual evidence over conversation or history.
- **Swappable VLM**: All model calls go through `app/pipeline/vlm_backend.py`. Swap Qwen2.5-VL for LLaVA or Gemma Vision by changing one config value.
- **Pydantic data models**: Every stage produces a typed Pydantic model, so bugs surface at stage boundaries not at output time.
- **Modular pipeline**: Each stage is a pure function `(input_model) → output_model`. Easy to test, swap, or run in parallel.
- **No hardcoded answers**: All decisions flow from VLM output + rule logic. No claim ID or image name is special-cased.

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# Process a batch:
python -m app.pipeline.runner --claims data/sample_claims/claims.csv \
       --history data/sample_history/history.csv \
       --images data/sample_images/ \
       --output output.csv
```

## Evaluation
```bash
python evaluation/evaluate.py --predictions output.csv --ground-truth evaluation/ground_truth.csv
```
