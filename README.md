# Grounded Optimization

**A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting**

Companion repository for the arXiv paper (link forthcoming). This repository contains:

- The **257-service cloud-provider contamination taxonomy** used by Layer 2
- **Evaluation scripts** that produce all tables in the paper
- **All raw result JSON files** (17 files across 16 experimental conditions, 680 LLM invocations)

---

## TL;DR

LLMs used to optimize personal documents (resumes, cover letters, applications) hallucinate in patterned ways: anachronistic technologies, cross-domain contamination of cloud-provider terminology, structural mutation of bullet content, and outright content fabrication. We characterize these as four failure modes (H1–H4) and propose a five-layer defense framework (L1–L5) combining deterministic detection, prompt-level grounding, and an evaluator agent.

**Headline findings:**

- Across 3 LLMs × 4 temperatures × 6 layer configurations, undefended baselines produce **2.48–5.36 detected hallucinations per resume**
- The full framework reduces detected hallucination rate to **0.04–0.24**
- Cross-domain contamination accounts for **79–89%** of baseline incidents and is the single dominant failure mode
- Temporal hallucinations are reduced by **50–95%** across configurations
- Prompt-level grounding alone (Layer 4) achieves zero detected hallucinations at low temperature with a strong model, but degrades at higher temperatures and on weaker models — motivating deterministic layers as a complement, not a replacement

All numbers in the paper are reproducible from the per-resume rates in `evaluation/results_comprehensive/`.

---

## Repository Structure

```
grounded-optimization/
├── paper-source/
│   ├── main.tex                    # LaTeX source of the arXiv paper
│   ├── references.bib              # Bibliography
│   └── preprint.sty                # NeurIPS-derived preprint style
├── taxonomy/
│   └── cloud_taxonomy.py           # 257-service taxonomy: AWS (76), GCP (53),
│                                   # Azure (64), On-Premise (64), plus 69
│                                   # cloud-agnostic technologies
├── evaluation/
│   ├── eval_comprehensive.py       # Main evaluation harness
│   ├── eval_deterministic_v2.py    # Deterministic detector implementation
│   ├── eval_phase2_e2e.py          # End-to-end pipeline evaluation
│   ├── eval_phase2_v2.py           # Phase 2 evaluator
│   └── results_comprehensive/      # All 17 raw JSON result files
├── CITATION.cff                    # Citation metadata
├── LICENSE                         # MIT
└── README.md
```

---

## The Five-Layer Defense (L1–L5)

| Layer | Purpose | Mechanism |
|-------|---------|-----------|
| **L1 — Temporal Context Validation** | Prevent anachronistic technology injection | Per-resume timeline + 30+ technology release dates embedded in agent prompts |
| **L2 — Cross-Domain Contamination Detection** | Catch cloud-provider bleeding | Deterministic, two-tier word-boundary regex over a 257-service taxonomy |
| **L3 — Structural Invariant Enforcement** | Prevent silent bullet/role compression | Pre/post counting of roles and bullet points with tolerance |
| **L4 — Prompt-Level Content Grounding** | First-line defense for fabrication | Explicit immutability rules for education, certifications, company names |
| **L5 — Evaluator Agent QA Gate** | Adversarial validation | Independent LLM critic that can reject and re-trigger the pipeline |

When validation fails, the system retries with augmented constraints. After 3 failed attempts, a deterministic fallback merge guarantees zero content loss.

---

## The Four Hallucination Modes (H1–H4)

| Mode | Definition | Example |
|------|-----------|---------|
| **H1 — Temporal Fabrication** | Reference to a technology that did not exist during the claimed time period | A 2018 role rewritten with "LangChain" (released late 2022) |
| **H2 — Cross-Domain Contamination** | Terminology from a foreign technology ecosystem injected into a role | An AWS-only role rewritten with "Azure Data Factory" |
| **H3 — Structural Mutation** | Silent reduction in bullets/roles, removing genuine accomplishments | 8 original bullets compressed to 4 generic summaries |
| **H4 — Content Fabrication** | Invented company names, inflated metrics, non-existent certifications | "Reduced API latency by 90%" added to a role with no performance numbers |

---

## Quickstart

### 1. Browse the contamination taxonomy

```python
from taxonomy.cloud_taxonomy import CLOUD_TAXONOMY, CLOUD_AGNOSTIC_TECHNOLOGIES

print(len(CLOUD_TAXONOMY))                          # 4 ecosystems
print(sum(len(v["services"]) for v in CLOUD_TAXONOMY.values()))  # 257 services
print(len(CLOUD_AGNOSTIC_TECHNOLOGIES))             # 69 cloud-agnostic terms
```

### 2. Reproduce the paper's numbers

```bash
cd evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install dependencies
cp .env.example .env              # add your OpenAI and Groq keys
python eval_comprehensive.py      # regenerates results/ (takes ~30 min)
```

### 3. Verify published numbers without re-running

Every aggregate in the paper's tables can be recomputed from the raw JSON files in
`evaluation/results_comprehensive/`:

| Paper reference | JSON file(s) |
|---|---|
| Table 1 (Ablation Study, GPT-4.1-nano, t=0) | `ablation_{baseline,prompt_only,L1_temporal,L1_L2,L1_L2_L3,full}.json` |
| Table 2 (Multi-Model Generalization) | `model_{gpt-4o-mini,llama-3.1-8b}_{baseline,full}.json` and ablation files for GPT-4.1-nano |
| Table 3 (Temperature Sensitivity) | `temp_{0.3,0.7,1.0}_{baseline,full}.json` |
| Cross-condition summary | `summary_20260418_181336.json` |

Each file records per-resume hallucination counts by type (H1, H2, H3, H4),
mean detection rate, standard deviation, 95% confidence interval, and the
exact `n_calls` invocation count.

---

## Honest Disclosures

The paper makes one important methodological disclosure that the reader should
keep in mind when looking at H2 contamination numbers: **the H2 detector and
the Layer 2 defense share the same `detect_role_contamination` function.**
When Layer 2 is active, contaminated output is reverted before the detector
runs, so H2 counts under defended configurations are mechanically zero by
construction. The large baseline H2 counts (measured without any active defense
and therefore free of this coupling) establish that the defense target is real,
but we cannot independently verify Layer 2 *eliminates* contamination versus
merely hiding it from our own detector. See Section 6.1 of the paper for the
full discussion. An independent NLI-based evaluator on the existing 680
outputs would close this gap.

---

## Citation

```bibtex
@article{indukuri2026grounded,
  title   = {Grounded Optimization: A Layered Engineering Framework for Reducing LLM Hallucination in Automated Personal Document Rewriting},
  author  = {Indukuri, Shashank},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://arxiv.org/abs/YYYY.NNNNN}
}
```

(Update the arXiv ID once the paper is live.)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

Questions, corrections, or extensions welcome via GitHub Issues or email:
**shashank.indukuri05@gmail.com**
