# PeptiScout AI — Cursor Master Context File
### CPS 5801 Advanced AI Systems | Kean University
### Version: Final | Use this file as context for every Cursor session

---

## ⚠️ CRITICAL INSTRUCTIONS FOR CURSOR

**Read this entire file before doing anything.**

This is a step-by-step implementation guide. You must build this project **in the exact task order defined in the BUILD ORDER section**. Do not skip ahead. Do not combine tasks. Each task depends on the previous one being complete and tested.

**Before starting any new task:**
1. Read the task section in full
2. Use PLAN mode to confirm your approach matches this document
3. Only then switch to AGENT mode to build

**Never:**
- Hardcode API keys anywhere (all keys come from `.env` via `python-dotenv`)
- Break existing endpoints when adding new ones
- Skip testing a tool before wiring it into the agent
- Run the fine-tuning notebook — that is done manually by the developer on Google Colab

---

## WHAT WE ARE BUILDING

**Project Name:** PeptiScout AI  
**What it is:** A reasoning-capable, multi-tool AI agent that helps peptide users get accurate information before buying or using peptides. It provides precise dosing math, mechanism-of-action (MOA) explanations at the pathway level, co-factor requirements, safety contraindications, PubMed-cited sources, and vendor legitimacy checks.

**The problem it solves:** General LLMs hallucinate clinical peptide details. Static dosing calculators have no reasoning. PeptiScout closes this gap with a multi-tool ReAct agent that calculates, retrieves, analyzes, and cites.

**Target users:**
- Biohackers and self-experimenters needing precise reconstitution math
- Fitness/bodybuilding community researching peptide protocols
- Clinical researchers needing MOA explanations with sourced citations

**This is also a university final project (CPS 5801) and must satisfy academic rubric requirements. The rubric requires three distinct system modes, an N=100 benchmark evaluation, and an ablation study. These are non-negotiable.**

---

## PARTS THE DEVELOPER IS DOING MANUALLY (DO NOT BUILD THESE)

The following are explicitly outside Cursor's scope. Do not generate code, stubs, or placeholders for these:

1. **IEEE Report** — developer writes this manually. Do not generate paper content.
2. **Presentation slides + recorded video** — out of scope entirely.
3. **Fine-tuning notebook execution** — Cursor writes the `.ipynb` file but does NOT run it. The developer uploads it to Google Colab (A100 GPU) and runs it there using the Colab plugin. After training, the developer downloads the LoRA adapter and places it at `/backend/models/peptide_lora_adapter/`.
4. **Building benchmark_100.json past the first 10 entries** — Cursor seeds the file with 10 example entries. The developer manually adds the remaining 90 queries with verified gold answers as each tool is completed.
5. **API key acquisition** — Developer creates accounts and gets keys for OpenAI, Pinecone, and Tavily. Keys go into `.env`. Cursor never touches real keys.
6. **THPdb CSV download** — Developer manually downloads the CSV from `thpdb.bicnirrh.res.in` and places it at `/backend/data/thpdb.csv` before the dataset generation script runs.
7. **Teammate contribution report** — developer writes a short paragraph manually.

---

## FULL TECH STACK

| Layer | Technology | Version / Notes |
|---|---|---|
| Frontend | React + Vite | Latest stable |
| Styling | Tailwind CSS | Utility classes only |
| State management | Zustand | Lightweight global store |
| HTTP client | Axios | Frontend → backend calls |
| Backend | FastAPI (Python) | Async, all API routes |
| Agent framework | LangGraph | Stateful ReAct graph |
| Baseline LLM | GPT-4o | Via OpenAI API (`gpt-4o`) |
| Fine-tuned LLM | Llama-3-8B + LoRA | `unsloth/llama-3-8b-bnb-4bit` |
| Fine-tune library | Unsloth + TRL + PEFT + bitsandbytes | Google Colab A100 only |
| Fine-tune format | Jupyter Notebook `.ipynb` | Required by academic rubric |
| VLM | GPT-4o-Vision | `gpt-4o` with image_url input |
| Vector database | Pinecone Serverless | Index name: `peptiscout` |
| Embeddings | text-embedding-3-small | OpenAI API |
| Web search | Tavily API | Vendor COA + Reddit vetting |
| Data source (RAG) | PubMed via NCBI E-Utils | No key required, email only |
| Data source (FT) | TDC via PyTDC package | `pip install PyTDC` |
| Data source (FT) | THPdb CSV | Manual download by developer |
| Dataset generation | GPT-4o as teacher model | Converts raw records to Alpaca pairs |
| Charts | Recharts | Results page ablation charts |
| PDF viewer | PDF.js | Report page (optional) |
| Deployment FE | Vercel | |
| Deployment BE | Railway or Render | |

---

## API KEYS & ENVIRONMENT

**File:** `/PeptiScout/.env` (never commit this)  
**File:** `/PeptiScout/.env.example` (commit this with placeholder values)

```
# OpenAI — GPT-4o baseline, GPT-4o-Vision, embeddings, dataset generation
OPENAI_API_KEY=your_key_here

# Pinecone — RAG vector database
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=peptiscout

# Tavily — vendor COA and Reddit source vetting
TAVILY_API_KEY=your_key_here

# NCBI — PubMed abstract fetching (email only, no key needed)
NCBI_EMAIL=your_email_here
```

**Backend reads keys with python-dotenv:**
```python
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

**NEVER pass OpenAI, Pinecone, or Tavily keys to the React frontend. All sensitive calls go through FastAPI only.**

---

## PROJECT FOLDER STRUCTURE

```
/PeptiScout
  /frontend                          # React + Vite app
    /src
      /components
      /pages
        Home.jsx
        Demo.jsx
        Architecture.jsx
        Results.jsx
        Team.jsx
      /store                         # Zustand state
      App.jsx
      main.jsx
    index.html
    vite.config.js
    tailwind.config.js
    package.json

  /backend                           # FastAPI app
    main.py                          # App entry point
    /routers
      query.py                       # POST /api/query (all modes)
      tools.py                       # POST /api/calculate, /api/analyze-bloodwork, /api/vetting
      results.py                     # GET /api/results
    /agent
      graph.py                       # LangGraph ReAct graph
    /tools
      calculator.py                  # Tool A — deterministic math
      rag_retriever.py               # Tool B — Pinecone semantic search
      vlm_analyzer.py                # Tool C — GPT-4o-Vision
      source_vetter.py               # Tool D — Tavily search
    /scripts
      generate_dataset.py            # Pulls TDC + THPdb, generates Alpaca pairs via GPT-4o
      ingest_pubmed.py               # Fetches PubMed abstracts, embeds, pushes to Pinecone
    /notebooks
      peptiscout_finetuning.ipynb    # LoRA fine-tuning notebook (run on Colab, NOT locally)
      peptiscout_evaluation.ipynb    # N=100 benchmark evaluation + ablation study
    /data
      thpdb.csv                      # Manually downloaded by developer
      peptide_dataset.json           # Generated by generate_dataset.py (5,000 Alpaca pairs)
      benchmark_100.json             # N=100 evaluation benchmark (seeded by Cursor, filled manually)
      results.json                   # Output of evaluation notebook, read by Results page
    /models
      /peptide_lora_adapter          # Downloaded from Colab after fine-tuning (manual step)
    requirements.txt
    .env
    .env.example
    .gitignore
```

---

## INPUT / OUTPUT SPECIFICATION

### What the system accepts
| Input Type | Format | Example |
|---|---|---|
| Peptide query | Natural language string | "What dose of BPC-157 for tendon repair?" |
| Lab report image | PNG or JPG upload | Bloodwork showing IGF-1, CRP values |
| Vendor name | Plain text in query | "Is Peptide Sciences legit?" |

### What the system always returns
A structured 4-part JSON response:
```json
{
  "protocol": "Dosing schedule with reconstitution math shown step by step",
  "moa": "Mechanism of action with specific pathway names (VEGF, NF-kB, etc.)",
  "good_bad": "Benefits and contraindications",
  "audit_trail": "PubMed citations with clickable PMIDs",
  "react_trace": "Thought/Action/Observation reasoning steps (full-agent mode only)"
}
```

---

## THE THREE SYSTEM MODES

These are required by the academic rubric. All three must be selectable from the Demo page UI.

| Mode | What It Is | Rubric Task |
|---|---|---|
| `baseline-zero-shot` | GPT-4o, no tools, no examples | Task 2 |
| `baseline-few-shot` | GPT-4o, no tools, 2 examples in prompt | Task 2 |
| `baseline-cot` | GPT-4o, no tools, chain-of-thought reasoning | Task 2 |
| `fine-tuned` | Llama-3-8B + LoRA adapter, no tools | Task 3.1 |
| `full-agent` | LangGraph ReAct + all 4 tools | Task 3.2 |

All 5 modes share the same endpoint: `POST /api/query`. The `mode` field in the request body switches execution paths.

---

## THE FOUR TOOLS (Task 3.2)

### Tool A — Deterministic Reconstitution Calculator
- **Type:** Pure Python function. Zero LLM involvement. 100% deterministic.
- **File:** `/backend/tools/calculator.py`
- **Endpoint:** `POST /api/calculate`
- **Why no LLM:** LLMs make ~40% math errors on syringe unit conversions. This tool is locked math — no hallucination possible.
- **Trigger condition:** Any query containing vial size, water volume, or dose numbers
- **Rubric category:** Planning module / deterministic tool

**Formula:**
```python
concentration_mcg_per_mL = (vial_mg * 1000) / water_mL
mL_to_inject = dose_mcg / concentration_mcg_per_mL
u100_units = mL_to_inject * 100
u40_units = mL_to_inject * 40
```

**Request:**
```json
{
  "vial_mg": 5,
  "water_mL": 2,
  "dose_mcg": 250,
  "syringe_type": "U100"
}
```

**Response:**
```json
{
  "concentration_mcg_per_mL": 2500,
  "inject_mL": 0.1,
  "syringe_units": 10,
  "label": "Draw 0.1mL = 10 units on a U100 syringe"
}
```

---

### Tool B — Clinical RAG (Retrieval-Augmented Generation)
- **Type:** Pinecone semantic search over 5,000 indexed PubMed abstracts
- **File:** `/backend/tools/rag_retriever.py`
- **NOT exposed as its own endpoint** — called internally by the LangGraph agent
- **Trigger condition:** Every single query. MOA and citations are always needed.
- **Rubric category:** Retrieval system ✅

**Two-phase implementation:**

Phase 1 — Ingestion (run once, offline via `ingest_pubmed.py`):
```
1. Query NCBI E-Utils API with peptide search terms
2. Fetch 5,000 abstract texts + PMID metadata
3. Chunk each abstract (~200 tokens per chunk)
4. Embed each chunk with text-embedding-3-small (OpenAI)
5. Upsert all vectors to Pinecone index "peptiscout" with PMID as metadata
```

Phase 2 — Retrieval (at query time):
```
1. Embed the user's query with text-embedding-3-small
2. Query Pinecone for top-3 most similar chunks
3. Return: chunk text + PMID + similarity score to synthesizer
```

**Input:** Query string (e.g., `"BPC-157 tendon repair VEGF co-factors"`)

**Output:**
```json
[
  {"pmid": "25627538", "text": "BPC-157 promotes tendon healing via VEGF...", "score": 0.91},
  {"pmid": "31205678", "text": "TB-500 synergy with BPC-157 angiogenic...", "score": 0.87},
  {"pmid": "28340951", "text": "Vitamin C co-factor for collagen synthesis...", "score": 0.83}
]
```

---

### Tool C — VLM Biomarker Analyzer
- **Type:** GPT-4o-Vision
- **File:** `/backend/tools/vlm_analyzer.py`
- **Endpoint:** `POST /api/analyze-bloodwork`
- **Trigger condition:** Image is attached to request OR query mentions a growth-factor peptide (BPC-157, Ipamorelin, CJC-1295, Sermorelin, GHRP-6)
- **Rubric category:** OCR / image analysis / VLM ✅

**Extraction prompt sent to GPT-4o-Vision:**
```
Extract the following biomarker values from this lab report image if present:
IGF-1 (ng/mL), CRP (mg/L), Total Testosterone (ng/dL), LH (mIU/mL), FSH (mIU/mL).
Return as JSON only. If a value is not visible or present, return "not found".
Do not return anything other than the JSON object.
```

**Request:**
```json
{
  "image_base64": "...",
  "image_type": "image/png"
}
```

**Response:**
```json
{
  "IGF-1": "187 ng/mL",
  "CRP": "1.2 mg/L",
  "Testosterone": "not found",
  "LH": "not found",
  "FSH": "not found",
  "flags": ["CRP mildly elevated — anti-inflammatory peptide protocol is appropriate"]
}
```

---

### Tool D — Source Vetting API
- **Type:** Tavily web search
- **File:** `/backend/tools/source_vetter.py`
- **Endpoint:** `POST /api/vetting`
- **Trigger condition:** Vendor name detected in query OR query contains phrases like "is X legit", "is X trustworthy", "should I buy from X"
- **Rubric category:** External API ✅

**Two Tavily searches fired per vetting request:**
1. `"{vendor} peptide COA certificate of analysis third party tested"`
2. `"site:reddit.com/r/Peptides {vendor} review reputation"`

**Request:**
```json
{
  "vendor_name": "Peptide Sciences"
}
```

**Response:**
```json
{
  "vendor": "Peptide Sciences",
  "coa_available": true,
  "coa_url": "https://...",
  "reddit_sentiment": "Generally positive — frequently recommended in r/Peptides",
  "flags": [],
  "summary": "COA available and third-party tested. Well-regarded in community."
}
```

---

## LANGGRAPH AGENT ARCHITECTURE (Task 3.2)

### ReAct Pattern
Before every tool call, the agent writes a visible trace:
```
Thought: [why this tool is needed right now]
Action: [which tool, with what parameters]
Observation: [what the tool returned]
Thought: [is more work needed, or can we synthesize?]
```

This full trace is returned as `react_trace` in the response JSON and shown in a collapsible panel in the Demo UI.

### Graph Nodes
| Node | File | Purpose |
|---|---|---|
| `router` | `graph.py` | Parses query, decides which tools to trigger |
| `proactive_check` | `graph.py` | If growth-factor peptide detected, flags that bloodwork is needed |
| `calculator` | calls `calculator.py` | Runs Tool A |
| `rag_retriever` | calls `rag_retriever.py` | Runs Tool B |
| `vlm_analyzer` | calls `vlm_analyzer.py` | Runs Tool C (only if image present) |
| `source_vetter` | calls `source_vetter.py` | Runs Tool D (only if vendor detected) |
| `synthesizer` | `graph.py` | Combines all tool outputs into the 4-part response |

### Graph Edges
```
router → proactive_check     (if growth-factor peptide in query)
router → calculator          (if dosage numbers in query)
router → rag_retriever       (always)
router → vlm_analyzer        (if image is attached)
router → source_vetter       (if vendor name in query)
all triggered tools → synthesizer
synthesizer → END
```

---

## BASELINE SYSTEM (Task 2)

All three baseline modes use GPT-4o with no tools, no fine-tuning, no retrieval.

### Mode: baseline-zero-shot
```
System prompt:
You are PeptiScout, an expert AI assistant specializing in research peptides.
When a user asks about a peptide, always respond in this exact structure:

[Protocol]: Provide dosing frequency, reconstitution instructions, and syringe math.
[MOA]: Explain the mechanism of action at the pathway level. Name the specific pathway.
[The Good / The Bad]: List primary benefits and known contraindications.
[The Audit Trail]: Cite at least 2 PubMed studies by PMID.

Be precise. Do not hedge excessively. Do not fabricate PMIDs.
```

### Mode: baseline-few-shot
Same system prompt as zero-shot, but prepend 2 worked examples before the user query:
```
EXAMPLE 1:
User: What is the reconstitution dose for 2mg of Semax with 1mL BAC water?
Assistant:
[Protocol]: 2mg in 1mL = 2000mcg/mL. Standard dose is 300mcg = 0.15mL = 15 units U100.
[MOA]: Semax is an ACTH analog that upregulates BDNF and NGF via TrkB receptor activation.
[The Good / The Bad]: Cognitive enhancement, neuroprotection. Contraindicated with MAOIs.
[The Audit Trail]: PMID 19230835, PMID 22750014.

EXAMPLE 2:
User: What co-factors does GHK-Cu require?
Assistant:
[Protocol]: GHK-Cu dose: 1-2mg/day subcutaneous or topical.
[MOA]: Tripeptide that downregulates IL-6 and TNF-α via NF-κB. Upregulates Collagen I/III.
[The Good / The Bad]: Anti-inflammatory, wound healing. Requires Vitamin C and Zinc.
[The Audit Trail]: PMID 25170290, PMID 28759605.
```

### Mode: baseline-cot
Same system prompt, but add before the response:
```
Before giving your structured response, think through the following inside <thinking> tags
(these will not be shown to the user):

<thinking>
1. What peptide is being asked about?
2. Does this query involve dosage math? If yes, calculate step by step.
3. Does this peptide require co-factors? List them.
4. Are there any contraindications I must flag?
5. Do I know verified PMIDs for this peptide? List only ones I am certain exist.
</thinking>

Then provide the [Protocol] / [MOA] / [Good-Bad] / [Audit Trail] response.
```

### Documented Baseline Failure Modes (for the rubric's "limitations" requirement)
| Failure Mode | Expected Error Rate |
|---|---|
| Syringe unit math errors (U40 vs U100) | ~40% |
| MOA lacks pathway-level specificity | ~60% |
| Fabricated / non-existent PMIDs | ~25% |
| Missing required co-factors | ~50% |
| Never proactively asks for bloodwork | ~0% |

---

## FINE-TUNING PIPELINE (Task 3.1)

### Goal
Adapt Llama-3-8B with LoRA so it describes peptide MOAs at the pathway/gene level instead of surface-level descriptions.

Before fine-tuning: *"GHK-Cu is a copper peptide used in skincare for anti-aging."*  
After fine-tuning: *"GHK-Cu downregulates IL-6 and TNF-α via NF-κB while upregulating Collagen I/III synthesis."*

### Dataset Generation (Cursor builds this, developer runs it)

**Script:** `/backend/scripts/generate_dataset.py`

**Pipeline:**
```
1. pip install PyTDC
2. Pull peptide records from TDC (Therapeutics Data Commons)
3. Load thpdb.csv from /backend/data/ (manually placed by developer)
4. For each raw record, call GPT-4o with this prompt:
   "Convert this raw peptide data into an Alpaca instruction-response pair.
    The instruction should ask about MOA and co-factors.
    The response MUST name specific pathways (NF-κB, VEGF, TrkB, etc.),
    specific cytokines or receptors, and list all required co-factors.
    Do not be vague. Return only valid JSON."
5. Collect 5,000 Alpaca pairs
6. Save to /backend/data/peptide_dataset.json
```

**Output format:**
```json
{
  "instruction": "Explain the mechanism and required co-factors for GHK-Cu.",
  "input": "",
  "output": "GHK-Cu is a tripeptide that downregulates IL-6 and TNF-α via the NF-κB pathway..."
}
```

**Note: PubMed abstracts are NOT used for fine-tuning. They go into Pinecone for RAG. These are separate pipelines.**

### Training Configuration (Cursor writes the notebook, developer runs it on Colab)

**Notebook:** `/backend/notebooks/peptiscout_finetuning.ipynb`

| Parameter | Value |
|---|---|
| Base model | `unsloth/llama-3-8b-bnb-4bit` |
| Method | LoRA |
| Library | Unsloth + TRL + PEFT |
| Hardware | Google Colab A100 (developer runs manually) |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Precision | fp16 |

**Required notebook sections:**
1. `## 1. Install Dependencies` — Unsloth, TRL, PEFT, bitsandbytes
2. `## 2. Load Base Model` — `unsloth/llama-3-8b-bnb-4bit`
3. `## 3. Load Dataset` — Alpaca JSON from `/backend/data/peptide_dataset.json`
4. `## 4. Apply LoRA Config` — exact values above
5. `## 5. Train` — SFTTrainer, 3 epochs
6. `## 6. Save Adapter` — save to `/peptide_lora_adapter/`
7. `## 7. Evaluate` — run 10 sample queries, show before/after comparison

**After Colab run:** Developer downloads `/peptide_lora_adapter/` and places it at `/backend/models/peptide_lora_adapter/`.

### Expected Results
| Metric | Baseline GPT-4o | Fine-Tuned Llama | Change |
|---|---|---|---|
| MOA Pathway Depth (PC) | ~40% | ~80% | +40pts ✅ |
| Co-factor Completeness | ~50% | ~85% | +35pts ✅ |
| Citation Accuracy (CA) | ~75% | ~40% | -35pts ⚠️ intentional — fine-tuning improves voice but not citations; this is why RAG is needed |

---

## EVALUATION & ABLATION STUDY (Task 4)

### The Benchmark Dataset

**File:** `/backend/data/benchmark_100.json`  
**Total entries:** 100  
**Cursor seeds the first 10. Developer fills the rest manually as each tool is built.**

**Entry schema:**
```json
{
  "id": 1,
  "query": "I have 5mg BPC-157 in 2mL BAC water. 80kg male. Dose for tendon healing?",
  "category": "dosage",
  "gold_mL": 0.1,
  "gold_cofactors": ["vitamin_c", "tb500"],
  "gold_pmids": ["25627538", "31205678"]
}
```

**Breakdown:**
- 40 entries: dosage/reconstitution math (`category: "dosage"`) — gold_mL is the correct answer
- 30 entries: MOA and co-factor queries (`category: "moa"`) — gold_cofactors is the correct answer
- 20 entries: safety/contraindication (`category: "safety"`) — scored via TSR binary rubric
- 10 entries: vendor vetting (`category: "vendor"`) — scored manually

**How developer fills the benchmark as tools are built:**
- After Tool A is done → add 40 dosage entries (calculator gives exact gold_mL)
- After RAG ingestion is done → add 30 MOA entries (pull co-factors from THPdb)
- Anytime → add 20 safety entries from PubMed case reports
- Anytime → add 10 vendor entries from Reddit r/Peptides threads

### Evaluation Notebook

**File:** `/backend/notebooks/peptiscout_evaluation.ipynb`

**What it does:**
1. Loads `benchmark_100.json`
2. Runs each query through all 3 system modes: baseline, fine-tuned, full-agent
3. Scores each response on DS, PC, TSR, CA
4. Runs all 3 ablations with modified system configurations
5. Exports `/backend/data/results.json` for the Results page

### Scoring Functions

```python
# DS — Dosage Success (exact math check)
ds_score = int(abs(predicted_mL - gold_mL) < 0.01)

# PC — Protocol Completeness (co-factor coverage)
required = entry["gold_cofactors"]
pc_score = sum(1 for c in required if c in response.lower()) / len(required)

# TSR — Task Success Rate (risk flagged AND solution provided)
tsr_score = int("contraindication" in response.lower() and "recommend" in response.lower())

# CA — Citation Accuracy (PMID actually exists in PubMed)
import requests
def validate_pmid(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}"
    r = requests.get(url)
    return "<e>" not in r.text  # NCBI returns error tag for invalid PMIDs

ca_score = all(validate_pmid(p) for p in extracted_pmids_from_response)
```

### Three-System Comparison Table (fill after running benchmark)
| Method | DS | PC | TSR | CA |
|---|---|---|---|---|
| Baseline GPT-4o Zero-Shot | TBD | TBD | TBD | TBD |
| Fine-Tuned Llama-3-8B + LoRA | TBD | TBD | TBD | TBD |
| Full Tool-Based Agent (LangGraph) | TBD | TBD | TBD | TBD |

### Ablation A — Remove Calculator (Tool A Disabled)
- **Implementation:** In `graph.py`, set `DISABLE_CALCULATOR = True`. Calculator node returns `None`. Synthesizer must attempt math unassisted.
- **Metric measured:** DS (Dosage Success)
- **Hypothesis:** DS drops >40%
- **What it proves:** Deterministic tools are required for safety math
- **Rubric maps to:** "removing planning module"

### Ablation B — Remove Reasoning (ReAct Disabled)
- **Implementation:** Add `mode="no-reasoning"` to endpoint. Bypasses LangGraph entirely. Query goes direct to GPT-4o with zero-shot prompt only.
- **Metric measured:** PC (Protocol Completeness)
- **Hypothesis:** PC drops >50%
- **What it proves:** Multi-step reasoning is what catches co-factors and edge cases
- **Rubric maps to:** "removing reasoning strategy"

### Ablation C — Fine-Tune Alone vs. RAG Alone
- **Implementation A:** `mode="fine-tuned-no-rag"` — load LoRA adapter, disable Pinecone call, no RAG chunks passed to synthesizer
- **Implementation B:** `mode="rag-only-no-finetune"` — GPT-4o + Pinecone, LoRA adapter not loaded
- **Metrics measured:** PC and CA side by side
- **Hypothesis:** Fine-tuning improves PC but hurts CA. RAG improves CA but doesn't fix PC. Full system fixes both.
- **What it proves:** Fine-tuning and RAG address different failure modes; the full system needs both
- **Rubric maps to:** "removing retrieval module"

---

## WEBSITE — 5 PAGES

### Design System
- **Fonts:** DM Sans (body) + Fraunces (headings) via Google Fonts
- **Colors:**
  - Background: `#FFFFFF` | Surface: `#F7F8FA`
  - Primary: `#0A6EFF` | Success: `#16A34A` | Danger: `#DC2626`
  - Text: `#0F172A` | Muted: `#64748B`
- **Style:** Rounded cards, subtle shadows, no gradients, clean dividers
- **Motion:** Fade-ins on load, smooth panel transitions on demo output

### Page 1: Home (`/`)
- Hero section with headline, subtext, two CTAs ("Try the Demo →" and "View Results →")
- Feature grid: 4 cards, one per tool with icon and description
- Static SVG of the agent architecture flowchart
- Sticky disclaimer footer: "For research purposes only. Not medical advice."

### Page 2: Demo (`/demo`)
**Two-column layout — left = input, right = output**

Input panel:
- Mode selector toggle: Baseline Zero-Shot / Baseline Few-Shot / Baseline CoT / Fine-Tuned / Full Agent
- Multi-line textarea for query
- Drag-and-drop image upload (PNG/JPG lab reports) for Tool C
- 3 clickable example queries that pre-fill the textarea
- "Analyze →" submit button (disabled while loading)

Output panel:
- Loading indicator (animated pulse)
- ReAct trace collapsible panel (Full Agent mode only)
- Protocol card (syringe math highlighted)
- MOA card (pathway terms bolded)
- Good/Bad card (green benefits / red risks, two columns)
- Audit Trail card (clickable PMIDs → PubMed in new tab)

### Page 3: Architecture (`/architecture`)
- Full agent flowchart (interactive SVG — clickable nodes expand with descriptions)
- 4 tool cards (model, trigger condition, input/output per tool)
- Full tech stack table
- Side-by-side comparison: Baseline vs. Fine-Tuned vs. Full Agent

### Page 4: Results (`/results`)
- Reads from `GET /api/results` → returns pre-computed `/backend/data/results.json`
- Hero stat cards for TSR, DS, PC, CA (full agent numbers)
- 3-system × 4-metric comparison table
- 3 Recharts bar charts — one per ablation with hypothesis label
- Pie chart showing benchmark category breakdown (40/30/20/10)

### Page 5: Team (`/team`)
- 2 team member cards with name, role, contribution breakdown
- 50% / 50% split displayed
- Tech stack logos

---

## FASTAPI ROUTE MAP

```
POST  /api/query              → mode switch: baseline-zero-shot | baseline-few-shot |
                                baseline-cot | fine-tuned | full-agent |
                                no-reasoning | fine-tuned-no-rag | rag-only-no-finetune
POST  /api/calculate          → Tool A standalone endpoint
POST  /api/analyze-bloodwork  → Tool C standalone endpoint
POST  /api/vetting            → Tool D standalone endpoint
GET   /api/results            → Returns pre-run results.json for Results page
GET   /api/health             → Health check
```

---

## BUILD ORDER — FOLLOW THIS EXACTLY, ONE STEP AT A TIME

**Do not skip steps. Do not combine steps. Each step must be built and tested before moving to the next.**

---

### STEP 0 — Scaffold Project Structure
**Rubric task:** Task 1  
**What to build:**
- Full folder structure as defined above
- `.env.example` with all placeholder keys
- `.gitignore` (ignores: `.env`, `__pycache__`, `node_modules`, `/backend/models/`)
- `requirements.txt` with all Python dependencies
- `package.json` for React frontend (React + Vite + Tailwind + Zustand + Axios + Recharts)
- FastAPI `main.py` with health check endpoint only
- Empty router and tool files (stubs only — no logic yet)

**Test:** `GET /api/health` returns `{"status": "ok"}`  
**Do not proceed until health check passes.**

---

### STEP 1 — Build Tool A: Reconstitution Calculator
**Rubric task:** Task 3.2 (Tool A)  
**What to build:**
- `/backend/tools/calculator.py` — the pure Python calculation function
- `POST /api/calculate` endpoint in `/backend/routers/tools.py`
- Input validation (vial_mg, water_mL, dose_mcg must all be positive numbers)

**Test with:**
```json
{"vial_mg": 5, "water_mL": 2, "dose_mcg": 250, "syringe_type": "U100"}
```
**Expected response:**
```json
{"concentration_mcg_per_mL": 2500, "inject_mL": 0.1, "syringe_units": 10, "label": "Draw 0.1mL = 10 units on a U100 syringe"}
```
**Do not proceed until this returns the exact correct values.**

---

### STEP 2 — Build Dataset Generation Script
**Rubric task:** Task 3.1  
**What to build:**
- `/backend/scripts/generate_dataset.py`
- Pulls peptide records from TDC using `PyTDC`
- Loads `/backend/data/thpdb.csv` (developer places this file manually first)
- Calls GPT-4o for each record to generate Alpaca instruction-response pairs
- Saves 5,000 pairs to `/backend/data/peptide_dataset.json`

**Developer action required before running:** Download THPdb CSV from `thpdb.bicnirrh.res.in` and place at `/backend/data/thpdb.csv`.

**Test:** Run script, confirm `peptide_dataset.json` is generated with correct Alpaca format. Spot-check 5 entries to confirm MOA responses name specific pathways.  
**Do not proceed until dataset file exists and spot-check passes.**

---

### STEP 3 — Build PubMed Ingestion Script (RAG Setup)
**Rubric task:** Task 3.2 (Tool B — Phase 1)  
**What to build:**
- `/backend/scripts/ingest_pubmed.py`
- Queries NCBI E-Utils API with peptide-specific search terms
- Fetches 5,000 PubMed abstracts with PMID metadata
- Chunks each abstract (~200 tokens)
- Embeds each chunk with `text-embedding-3-small`
- Upserts all vectors to Pinecone index `peptiscout` with PMID stored as metadata

**Test:** After ingestion, query Pinecone manually with `"BPC-157 tendon repair"`. Confirm top-3 results return with real PMIDs attached.  
**Do not proceed until Pinecone returns valid results for a test query.**

---

### STEP 4 — Build Baseline LLM Endpoint
**Rubric task:** Task 2  
**What to build:**
- `POST /api/query` in `/backend/routers/query.py`
- Supports three modes: `baseline-zero-shot`, `baseline-few-shot`, `baseline-cot`
- Each mode calls GPT-4o with the appropriate system prompt as defined in this document
- Returns structured 4-part JSON response
- No LangGraph, no tools, no fine-tuning — GPT-4o direct call only

**Test with:**
```json
{"text": "I have 5mg BPC-157 and 2mL BAC water. 80kg. Dose for tendon healing?", "mode": "baseline-zero-shot"}
```
Confirm response contains all 4 keys: `protocol`, `moa`, `good_bad`, `audit_trail`.  
Test all 3 modes. **Do not proceed until all 3 modes return valid structured responses.**

---

### STEP 5 — Build Fine-Tuning Notebook
**Rubric task:** Task 3.1  
**What to build:**
- `/backend/notebooks/peptiscout_finetuning.ipynb`
- 7 required sections as defined in the Fine-Tuning section above
- Uses exact LoRA hyperparameters from this document
- Loads dataset from `/backend/data/peptide_dataset.json`
- Saves adapter to `/peptide_lora_adapter/`
- Section 7 runs 10 sample queries and shows before/after comparison

**CURSOR DOES NOT RUN THIS NOTEBOOK.**  
Developer uploads to Google Colab, runs on A100, downloads adapter, places at `/backend/models/peptide_lora_adapter/`.

**Test (after developer runs on Colab and downloads adapter):** Load adapter locally, run 3 sample queries, confirm MOA responses now name specific pathways.

---

### STEP 6 — Wire Fine-Tuned Mode into /api/query
**Rubric task:** Task 3.1  
**What to build:**
- Add `mode="fine-tuned"` to the existing `POST /api/query` endpoint
- Loads LoRA adapter from `/backend/models/peptide_lora_adapter/`
- Runs inference with the fine-tuned Llama-3-8B model
- Returns same 4-part JSON structure as baseline modes
- Do NOT break existing baseline modes

**Test:** Run the same BPC-157 query in fine-tuned mode. Confirm MOA response names specific pathways (NF-κB, VEGF, etc.) unlike the baseline.  
**Do not proceed until fine-tuned mode returns pathway-level MOA responses.**

---

### STEP 7 — Build Tools B, C, D + Wire into Standalone Endpoints
**Rubric task:** Task 3.2 (Tools B, C, D)  
**What to build:**

Tool B (`/backend/tools/rag_retriever.py`):
- Takes query string, embeds with text-embedding-3-small, queries Pinecone, returns top-3 chunks + PMIDs
- Test: query `"BPC-157 VEGF tendon"`, confirm 3 chunks returned with real PMIDs

Tool C (`/backend/tools/vlm_analyzer.py` + `POST /api/analyze-bloodwork`):
- Accepts base64 image, sends to GPT-4o-Vision with extraction prompt, returns biomarker JSON
- Test: send a sample lab image, confirm structured biomarker values returned

Tool D (`/backend/tools/source_vetter.py` + `POST /api/vetting`):
- Accepts vendor name, fires 2 Tavily searches, returns vetting summary
- Test: send `{"vendor_name": "Peptide Sciences"}`, confirm Tavily results are summarized

**Do not proceed until all three tools return valid responses independently.**

---

### STEP 8 — Build LangGraph Full Agent
**Rubric task:** Task 3.2  
**What to build:**
- `/backend/agent/graph.py` — LangGraph stateful ReAct graph
- All 7 nodes: router, proactive_check, calculator, rag_retriever, vlm_analyzer, source_vetter, synthesizer
- All edges as defined in the architecture section
- Add `mode="full-agent"` to `POST /api/query`
- ReAct trace (Thought/Action/Observation) returned as `react_trace` field in response
- Do NOT break existing baseline or fine-tuned modes

**Ablation flags to include in graph.py:**
```python
DISABLE_CALCULATOR = False   # Set True for Ablation A
DISABLE_REACT = False        # Set True for Ablation B
DISABLE_RAG = False          # Set True for Ablation C variant
DISABLE_FINETUNE = False     # Set True for Ablation C variant
```

**Test with:** 
```json
{
  "text": "I have 5mg BPC-157 in 2mL BAC water. 80kg. Tendon healing. Thinking of stacking TB-500.",
  "mode": "full-agent",
  "image_base64": null
}
```
Confirm: `react_trace` is populated, `audit_trail` contains real PMIDs, `protocol` contains syringe math.  
**Do not proceed until full agent returns a valid 4-part response with react_trace.**

---

### STEP 9 — Build Evaluation Notebook + Ablations
**Rubric task:** Task 4  
**What to build:**
- `/backend/notebooks/peptiscout_evaluation.ipynb`
- Loads `benchmark_100.json` (must be complete by this point)
- Runs all 100 queries through all modes: baseline, fine-tuned, full-agent
- Scores each with DS, PC, TSR, CA scoring functions defined in this document
- Runs Ablation A (DISABLE_CALCULATOR=True), measures DS delta
- Runs Ablation B (mode="no-reasoning"), measures PC delta
- Runs Ablation C (fine-tuned-no-rag vs rag-only-no-finetune), measures PC and CA
- Exports all results to `/backend/data/results.json`

Also add to `POST /api/query`:
- `mode="no-reasoning"` (Ablation B — direct GPT-4o, no LangGraph)
- `mode="fine-tuned-no-rag"` (Ablation C variant A)
- `mode="rag-only-no-finetune"` (Ablation C variant B)

**Test:** Run notebook on the first 10 benchmark entries. Confirm `results.json` is generated with correct structure.

---

### STEP 10 — Build React Frontend (All 5 Pages)
**Rubric task:** Website  
**What to build:**
- React + Vite app with React Router for 5 routes
- Design system implemented (fonts, colors, card styles as defined above)
- Build in order: Home → Demo → Results → Architecture → Team

**Demo page connects to FastAPI at `localhost:8000`.**  
**Results page fetches from `GET /api/results` — does not hardcode data.**

**Test:** Run frontend against live backend. Submit the BPC-157 example query in Full Agent mode. Confirm all 4 output panels render correctly. Confirm ReAct trace collapsible works.

---

### STEP 11 — Connect, Test End-to-End, Deploy
**What to build:**
- Connect all frontend pages to backend
- Test every mode in Demo page
- Test Results page with real `results.json` data
- Deploy frontend to Vercel
- Deploy backend to Railway or Render
- Update frontend Axios base URL to point to deployed backend

**Final test:** Run the full BPC-157 scenario end-to-end on the deployed system.

---

## EXAMPLE FULL AGENT EXECUTION TRACE

This is what happens internally when a user submits:
> *"I have a 5mg vial of BPC-157 reconstituted with 2mL BAC water. I weigh 80kg and want to use it for tendon repair. Thinking about stacking with TB-500."* + bloodwork image

```
Thought: Query has vial (5mg), water (2mL), weight (80kg) → Tool A must fire
         Query asks about tendon repair + TB-500 stack → Tool B must fire
         Image is attached → Tool C must fire
         No vendor name → Tool D does NOT fire
         BPC-157 is growth-factor peptide → proactive_check fires

Action: calculator(vial_mg=5, water_mL=2, dose_mcg=250, syringe_type="U100")
Observation: {inject_mL: 0.1, syringe_units: 10, concentration: 2500 mcg/mL}

Action: rag_retriever("BPC-157 tendon repair VEGF co-factors TB-500 synergy")
Observation: [
  {pmid: "25627538", text: "BPC-157 upregulates VEGF and GH receptor..."},
  {pmid: "31205678", text: "BPC-157 + TB-500 additive angiogenic effects..."},
  {pmid: "28340951", text: "Vitamin C co-factor for collagen synthesis..."}
]

Action: vlm_analyzer(image_base64="...")
Observation: {IGF-1: "187 ng/mL", CRP: "1.2 mg/L", flags: ["CRP mildly elevated — BPC-157 anti-inflammatory profile appropriate"]}

Thought: All needed tools have returned. IGF-1 is normal — no protocol modification.
         CRP is mildly elevated — flag this as supporting rationale.
         I have syringe math, RAG chunks with PMIDs, and bloodwork context.
         Ready to synthesize.

Action: synthesizer(calculator_result, rag_chunks, vlm_result)

Final Response:
{
  "protocol": "5mg in 2mL BAC water = 2,500 mcg/mL. 250mcg dose = 0.1mL = 10 units U100. 
               Inject subcutaneously near injury site, once daily. Stack TB-500 at 2mg 2x/week.
               Co-factors: Vitamin C 500mg/day required for collagen synthesis.
               Your CRP is mildly elevated (1.2 mg/L) — BPC-157 anti-inflammatory profile is appropriate.",
  "moa": "BPC-157 upregulates VEGF (angiogenesis) and GH receptor expression at injury site. 
          TB-500 promotes actin polymerization via thymosin-β4. Together: vascular + structural repair.",
  "good_bad": "✅ Accelerated tendon healing, anti-inflammatory, angiogenic\n✅ TB-500 stack is additive — different pathway\n⚠️ Avoid if history of hormone-sensitive conditions\n⚠️ Retest IGF-1 at 30 days",
  "audit_trail": "PMID 25627538 — BPC-157 VEGF/GH upregulation in tendon healing\nPMID 31205678 — BPC-157 + TB-500 additive angiogenic effects\nPMID 28340951 — Vitamin C co-factor for collagen synthesis",
  "react_trace": "Thought: ... Action: calculator... Observation: ... Action: rag_retriever..."
}
```

---

## QUICK REFERENCE — DATA FLOWS

```
TDC + THPdb → generate_dataset.py → peptide_dataset.json → finetuning.ipynb → LoRA adapter
                                                                    ↑
                                                    (runs on Colab, manual step)

PubMed (NCBI E-Utils) → ingest_pubmed.py → Pinecone index "peptiscout"
                                                    ↓
                                            rag_retriever.py (at query time)

benchmark_100.json → peptiscout_evaluation.ipynb → results.json → Results page (/api/results)

User query → POST /api/query → mode switch → LangGraph (full-agent) or GPT-4o direct (baseline) → 4-part JSON response
```

---

## REQUIREMENTS CHECKLIST (Academic Rubric)

Use this to confirm nothing is missed before submission:

- [ ] Task 1: Problem + target users defined
- [ ] Task 1: Input/output format defined
- [ ] Task 1: Task type defined
- [ ] Task 1: Capabilities and limitations documented
- [ ] Task 1: Agent architecture flowchart (in Architecture page of website)
- [ ] Task 1: Evaluation metrics defined (DS, PC, TSR, CA, F1)
- [ ] Task 1: Benchmark dataset defined (N=100, 4 categories)
- [ ] Task 2: Zero-shot baseline working
- [ ] Task 2: Few-shot baseline working
- [ ] Task 2: Chain-of-thought baseline working
- [ ] Task 2: Baseline evaluation results filled in (after benchmark run)
- [ ] Task 2: Limitations documented ("The Wall" — 5 failure modes)
- [ ] Task 3.1: Dataset constructed (5,000 Alpaca pairs)
- [ ] Task 3.1: Training setup documented (LoRA hyperparameters)
- [ ] Task 3.1: Fine-tuning notebook built and run on Colab
- [ ] Task 3.1: Fine-tuned mode wired into /api/query
- [ ] Task 3.1: Improvement over baseline measured and documented
- [ ] Task 3.2: Tool A — Calculator working
- [ ] Task 3.2: Tool B — RAG working (Pinecone populated)
- [ ] Task 3.2: Tool C — VLM working
- [ ] Task 3.2: Tool D — Tavily working
- [ ] Task 3.2: LangGraph multi-step reasoning working
- [ ] Task 3.2: ReAct trace visible in response and UI
- [ ] Task 4: All 3 systems compared on N=100 benchmark
- [ ] Task 4: Ablation A complete (calculator removed, DS measured)
- [ ] Task 4: Ablation B complete (reasoning removed, PC measured)
- [ ] Task 4: Ablation C complete (FT-only vs RAG-only, PC and CA measured)
- [ ] Task 4: results.json exported and Results page populated
- [ ] Website: All 5 pages built and connected to backend
- [ ] Website: Demo page works end-to-end for all 5 modes
- [ ] Teammate contribution report written (manual — 50/50 split)
