## Entity & Relation Extraction from PDF (Open-source LLM)

This project extracts entities and relations from a PDF using a local open-source LLM (Mistral 7B Instruct via llama.cpp) plus high-precision regex for PAN detection.

* Entities: Organisation, Name (person), PAN
* Relation: PAN_Of (PAN belongs to Person)

### Prerequisites

* Python 3.10+
* Sufficient RAM/CPU. GPU optional (set `LLAMA_N_GPU_LAYERS` for partial offload)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python extract_entities.py --pdf "PDF for Python LLM.pdf" --out output
```

The first run downloads a quantized GGUF: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf`.

### Outputs (`--out`)

* `entities.csv` with columns: `id,type,text`
* `relations.csv` with columns: `type,head,tail` (heads/tails reference `entities.csv` `id`)
* `pan_of.csv` with columns: `PAN,Relation,Person` (directly usable)

### Notes for Quality

* PANs are first detected via regex and only these are trusted.
* LLM proposes Names/Organisations and candidate relations.
* Reconciliation keeps only relations where PAN is regex-confirmed and Person is a detected Name.
* The prompt prioritizes precision to reduce false positives.

### Custom Model

Override with:

```bash
python extract_entities.py --pdf "PDF for Python LLM.pdf" \
  --model-repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --model-file mistral-7b-instruct-v0.2.Q4_K_M.gguf
```


