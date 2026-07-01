# Evaluation — How to Reproduce the Paper's Results

This directory contains everything needed to regenerate Tables 1–3 from the paper.

## Setup

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add GROQ_API_KEY (required) and OPENAI_API_KEY (required only for OpenAI models)
```

API keys:
- **Groq** — free tier: https://console.groq.com/keys
- **OpenAI** — only needed if reproducing `gpt-4o-mini` / `gpt-4.1-nano` conditions

## Running

### Full comprehensive sweep (Tables 1–3)

```bash
python eval_comprehensive.py
```

- Duration: ~30 minutes
- Writes per-condition JSON files to `results/` (will overwrite existing)
- Writes an aggregate `summary_TIMESTAMP.json`

Three experimental axes:
1. **Temperature sweep** (`gpt-4.1-nano`, T ∈ {0.3, 0.7, 1.0}) — Table 1
2. **Model sweep** (`gpt-4o-mini`, `llama-3.1-8b` via Groq, T=0.0) — Table 2
3. **Ablation** (baseline / prompt-only / L1+temporal / L1+L2 / L1+L2+L3 / full L1+L2+L3+L4+L5) — Table 3

Each condition evaluates 25 resumes × 2 configs (baseline vs full) and counts H1–H4 hallucinations per output.

### End-to-end pipeline eval (§4.4)

```bash
python eval_phase2_e2e.py
```

End-to-end generation pipeline exercising all 5 defense layers on real resumes. Used for the qualitative examples in Appendix E.

## Interpreting the JSON files

Each result file has this shape:

```json
{
  "<condition_name>": {
    "model": "gpt-4.1-nano",
    "config": "baseline",
    "temperature": 0.3,
    "n_resumes": 25,
    "total_halls": {"H1": 7, "H2": 55, "H3": 0, "H4": 0, "total": 62, "errors": 0},
    "mean_hr": 2.48,
    "std_hr": 3.8379,
    "ci_95": 1.5045,
    "resume_rates": [17, 2, 0, 0, 4, ...]
  }
}
```

| Field | Meaning |
|-------|---------|
| `total_halls.{H1,H2,H3,H4}` | Total hallucinations of each category across all 25 resumes |
| `mean_hr` | Mean hallucinations per resume |
| `std_hr`  | Standard deviation across resumes |
| `ci_95`   | 95% confidence interval half-width |
| `resume_rates` | Per-resume hallucination count (length 25) |

## Mapping JSON files to paper tables

### Table 1 — Temperature sensitivity (`gpt-4.1-nano`)

| Condition | Baseline file | Full file |
|-----------|---------------|-----------|
| T = 0.3   | `temp_0.3_baseline.json` | `temp_0.3_full.json` |
| T = 0.7   | `temp_0.7_baseline.json` | `temp_0.7_full.json` |
| T = 1.0   | `temp_1.0_baseline.json` | `temp_1.0_full.json` |

### Table 2 — Model generalization

| Model | Baseline file | Full file |
|-------|---------------|-----------|
| `gpt-4o-mini`  | `model_gpt-4o-mini_baseline.json`  | `model_gpt-4o-mini_full.json` |
| `llama-3.1-8b` (Groq) | `model_llama-3.1-8b_baseline.json` | `model_llama-3.1-8b_full.json` |

### Table 3 — Layer ablation (`gpt-4.1-nano`, T=0.0)

| Config | File |
|--------|------|
| Baseline (no defenses) | `ablation_baseline.json` |
| Prompt-only (no layers, just improved prompting) | `ablation_prompt_only.json` |
| L1 + temporal grounding | `ablation_L1_temporal.json` |
| L1 + L2 | `ablation_L1_L2.json` |
| L1 + L2 + L3 | `ablation_L1_L2_L3.json` |
| Full (L1 + L2 + L3 + L4 + L5) | `ablation_full.json` |

## Notes on reproducibility

- **Non-determinism:** LLM outputs are not exactly reproducible even at T=0.0 across API calls (provider-side non-determinism). Expected deviation from published numbers: ±1–2 hallucinations per condition.
- **Model availability:** `gpt-4.1-nano` is the OpenAI model used. If it is deprecated, substitute `gpt-4o-mini` — results in the paper include both.
- **Groq rate limits:** Free tier has per-minute token caps. The script paces requests but may need a retry on 429.

## Questions / issues

Open a GitHub issue or email `sinduku1@depaul.edu`.
