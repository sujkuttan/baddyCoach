# Gemini API Setup for BaddyCoach

## What Gemini Does

The Gemini integration provides AI-powered coaching narration that:

- **Summarizes the match** with natural-language observations
- **References rule-based findings** (strengths, weaknesses from the YAML rule engine)
- **Grounds every claim** in actual metrics — no hallucinated numbers
- Runs via the `shuttle_coach` subsystem when `GEMINI_API_KEY` is set

## Getting an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key (starts with `AIza...`)

## Where to Place the Key

### Option 1: Environment Variable (Recommended)

```bash
export GEMINI_API_KEY="AIza..."
```

Add this to your `.bashrc`, `.zshrc`, or the terminal session where you run BaddyCoach:

```bash
echo 'export GEMINI_API_KEY="AIza..."' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: `.env` File

Create a `.env` file in the project root (`/home/sujith/baddyCoach/.env`):

```
GEMINI_API_KEY=AIza...
```

The app reads it via `settings.py` (`gemini_api_key: str | None = None`), which inherits from `pydantic_settings.BaseSettings` and automatically loads `.env` files.

### Option 3: Pass at Runtime

The `GEMINI_API_KEY` environment variable is checked at pipeline runtime in `routes.py:_generate_narration()`. Either form works:

```python
import os
os.environ["GEMINI_API_KEY"] = "AIza..."
```

## How It Works

```
Pipeline completes
  → rules.yaml evaluated (25+ threshold-based rules)
  → shuttle_coach.analyze() runs (shot effectiveness, movement, errors)
  → Gemini narration generated with:
      1. Metrics as grounded evidence
      2. Rule-based findings as conversational context
  → narration added to report["narration"]
```

### Narration Example

> *"Your smash effectiveness is strong at 58%, but your clear win rate of 22% suggests opponents are exploiting deep returns. Focus on varying clear height and adding a drop option from the rear court. Recovery time of 1.2s leaves you vulnerable — the 0.8s threshold for quick reset is achievable with split-step drills."*

Every claim references a metric ID in square brackets — citations are enforced server-side.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No narration in report | API key not set | Set `GEMINI_API_KEY` env var |
| `google-generativeai` import error | Package not installed | `pip install google-generativeai` |
| "Ungrounded sentences" error | LLM hallucinated | The narration module rejects it; try a different prompt |
| "Cited unknown metrics" error | Metric ID mismatch | Check metric IDs in `shuttle_coach/metrics/` |

## Model

Default model: `gemini-2.0-flash` (configured in `shuttle_coach/narration/gemini.py:32`)

You can change it to `gemini-2.0-pro` or `gemini-1.5-pro` for higher quality (slower, more expensive).
