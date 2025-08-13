import os
import re
import json
import argparse
from typing import List, Dict, Tuple, Optional

import pdfplumber
import pandas as pd
from tqdm import tqdm
from string import Template

# Use regex (the `regex` module is installed but we can rely on re here)
PAN_REGEX = re.compile(r"\b[A-Z]{5}[A-Z0-9]{4}[A-Z]\b")
NAME_HONORIFICS = r"(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Shri|Smt\.?|M/s\.?|Sri)"
NAME_CANDIDATE_REGEX = re.compile(
    rf"\b(?:{NAME_HONORIFICS}\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
)

SYSTEM_PROMPT = (
    "You are an information extraction assistant.\n"
    "Extract entities and relations from the given text.\n"
    "Entities to extract: Organisation, Name (person), PAN.\n"
    "Relation to extract: PAN_Of (PAN belongs to Person).\n"
    "Rules:\n"
    "- Prefer high precision over recall; avoid hallucinations.\n"
    "- Only output entities that appear explicitly in the text.\n"
    "- A PAN is a 10-character Indian PAN-like code (e.g., ABCDE1234F).\n"
    "- Names: honorifics like Mr., Ms., Mrs., Dr. can appear.\n"
    "- Organisation: company/firm/LLP/authority names.\n"
    "Output strict JSON with keys: entities, relations.\n"
    "Schema:\n"
    "{\n  'entities': [\n    {'id': 'e1', 'type': 'PAN|Name|Organisation', 'text': '...'},\n    ...\n  ],\n  'relations': [\n    {'type': 'PAN_Of', 'head': 'ePAN or PAN text', 'tail': 'ePerson or Name text'}\n  ]\n}\n"
)

USER_PROMPT_TEMPLATE = (
    "Extract entities and relations strictly following the schema.\n"
    "Text:\n" +
    """\n{chunk}\n""" +
    "\nReturn only JSON."
)

# Focused prompt for a single PAN within a small local context
FOCUSED_PROMPT_TEMPLATE = Template(
    "You will be given a short text and a specific PAN to verify.\n"
    "Task:\n"
    "1) List explicitly mentioned Names and Organisations in the text.\n"
    "2) If and only if the provided PAN is present in the text AND it belongs to a specific person, return a PAN_Of relation mapping that PAN to the correct person.\n"
    "3) Be precise. If uncertain, return no relations.\n\n"
    "Return strict JSON with keys: entities, relations.\n"
    "- For entities, use types: 'Name', 'Organisation', 'PAN'.\n"
    "- For relations, use 'PAN_Of' with head as the PAN text and tail as the Name text.\n\n"
    "Example:\n"
    "{\n  'entities': [\n    {'type': 'PAN', 'text': 'ABCDE1234F'},\n    {'type': 'Name', 'text': 'Mr. Agarwal'},\n    {'type': 'Organisation', 'text': 'Acme Pvt Ltd'}\n  ],\n  'relations': [\n    {'type': 'PAN_Of', 'head': 'ABCDE1234F', 'tail': 'Mr. Agarwal'}\n  ]\n}\n\n"
    "PAN: $pan\n"
    "Text:\n$context\n"
)


def load_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def find_pans(text: str) -> List[str]:
    return sorted(set(PAN_REGEX.findall(text)))


# LLM via llama.cpp python binding
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


MODEL_PRESETS = {
    # Good quality, very light
    "qwen1_5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    # Larger but still light enough on CPU
    "qwen3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
    },
    # Heavier; previous default
    "mistral7b": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
    # Ultra light option
    "tinyllama": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    },
}


def ensure_model(model_repo: str, filename: str) -> str:
    # download into HF cache and return local path; try exact, if fails, attempt common mirrors
    try:
        return hf_hub_download(repo_id=model_repo, filename=filename)
    except Exception:
        mirrors = [
            (model_repo, filename),
            ("TheBloke/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct.Q4_K_M.gguf"),
            ("TheBloke/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct.Q4_K_M.gguf"),
        ]
        last_err = None
        for repo, file in mirrors:
            try:
                return hf_hub_download(repo_id=repo, filename=file)
            except Exception as e:
                last_err = e
                continue
        raise last_err


def get_llm(model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None) -> Llama:
    if n_threads is None:
        try:
            n_threads = max(1, os.cpu_count() or 4)
        except Exception:
            n_threads = 4
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        verbose=False,
    )
    return llm


def llm_extract_chunk(llm: Llama, chunk: str, temperature: float = 0.0, max_tokens: int = 512) -> Dict:
    prompt = f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n{USER_PROMPT_TEMPLATE.format(chunk=chunk)} [/INST]"
    output = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "[/INST]"],
    )
    text = output["choices"][0]["text"].strip()
    # try to isolate JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Not a dict JSON")
        data.setdefault("entities", [])
        data.setdefault("relations", [])
        return data
    except Exception:
        return {"entities": [], "relations": []}


def llm_extract_for_pan(llm: Llama, pan: str, context: str, temperature: float = 0.0, max_tokens: int = 384) -> Dict:
    prompt = FOCUSED_PROMPT_TEMPLATE.substitute(pan=pan, context=context)
    output = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "[/INST]"],
    )
    text = output["choices"][0]["text"].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Not a dict JSON")
        data.setdefault("entities", [])
        data.setdefault("relations", [])
        # ensure PAN entity included for head resolution
        if pan not in [e.get("text") for e in data["entities"] if e.get("type") == "PAN"]:
            data["entities"].append({"type": "PAN", "text": pan})
        return data
    except Exception:
        return {"entities": [{"type": "PAN", "text": pan}], "relations": []}


def chunk_text(pages: List[Tuple[int, str]] , max_chars: int = 2500) -> List[Tuple[List[int], str]]:
    chunks: List[Tuple[List[int], str]] = []
    current_pages: List[int] = []
    current_text: List[str] = []
    current_len = 0
    for page_num, text in pages:
        if not text:
            continue
        if current_len + len(text) > max_chars and current_text:
            chunks.append((current_pages.copy(), "\n".join(current_text)))
            current_pages = []
            current_text = []
            current_len = 0
        current_pages.append(page_num)
        current_text.append(text)
        current_len += len(text)
    if current_text:
        chunks.append((current_pages, "\n".join(current_text)))
    return chunks


def build_pan_contexts(pages: List[Tuple[int, str]], pans: List[str], window_chars: int = 1200) -> Dict[str, List[Tuple[int, str]]]:
    pan_to_contexts: Dict[str, List[Tuple[int, str]]] = {p: [] for p in pans}
    for page_num, text in pages:
        for match in PAN_REGEX.finditer(text or ""):
            pan = match.group(0)
            if pan not in pan_to_contexts:
                continue
            start = max(0, match.start() - window_chars)
            end = min(len(text), match.end() + window_chars)
            context = text[start:end]
            pan_to_contexts[pan].append((page_num, context))
    return pan_to_contexts


def is_likely_person_name(name: str) -> bool:
    s = name.strip()
    if not s or len(s) > 100:
        return False
    if not re.match(r"^[A-Za-z .,'-]+$", s):
        return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    if len(tokens) < 1 or len(tokens) > 5:
        return False
    # skip organisation-like suffixes
    org_suffixes = {"Ltd", "Limited", "Pvt", "Private", "LLP", "Inc", "LLC", "Company", "Co"}
    if any(t in org_suffixes for t in tokens):
        return False
    # At least one capitalized word
    if not any(t[0:1].isupper() for t in tokens if t.isalpha()):
        return False
    return True


def heuristic_names_from_context(context: str) -> List[str]:
    candidates: List[str] = []
    for m in NAME_CANDIDATE_REGEX.finditer(context):
        full = m.group(0).strip()
        if is_likely_person_name(full):
            candidates.append(full)
    # preserve order, dedup
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def reconcile(pan_candidates: List[str], llm_json: Dict) -> Tuple[List[Dict], List[Dict]]:
    entities: List[Dict] = []
    relations: List[Dict] = []

    id_counter = 1
    pan_text_to_id: Dict[str, str] = {}
    name_text_to_id: Dict[str, str] = {}
    org_text_to_id: Dict[str, str] = {}

    # Add PANs from regex as authoritative
    for pan in pan_candidates:
        ent_id = f"e{id_counter}"
        id_counter += 1
        entities.append({"id": ent_id, "type": "PAN", "text": pan})
        pan_text_to_id[pan] = ent_id

    # Add LLM entities (only Names and Organisations; PANs only if confirmed)
    for ent in llm_json.get("entities", []):
        ent_type = (ent.get("type") or "").strip()
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        if ent_type.upper() == "PAN":
            if text in pan_text_to_id:
                pass
            else:
                continue
        elif ent_type.lower() in {"name", "person", "person_name"}:
            if text not in name_text_to_id and is_likely_person_name(text):
                ent_id = f"e{id_counter}"
                id_counter += 1
                entities.append({"id": ent_id, "type": "Name", "text": text})
                name_text_to_id[text] = ent_id
        elif ent_type.lower() in {"organisation", "organization", "company", "org"}:
            if text not in org_text_to_id:
                ent_id = f"e{id_counter}"
                id_counter += 1
                entities.append({"id": ent_id, "type": "Organisation", "text": text})
                org_text_to_id[text] = ent_id

    # Relations: accept only PAN_Of where head is confirmed PAN and tail is a Name (create if missing)
    for rel in llm_json.get("relations", []):
        if rel.get("type") != "PAN_Of":
            continue
        head = rel.get("head")
        tail = rel.get("tail")
        head_id = None
        tail_id = None
        if isinstance(head, str) and head in pan_text_to_id:
            head_id = pan_text_to_id[head]
        if isinstance(tail, str):
            if tail in name_text_to_id:
                tail_id = name_text_to_id[tail]
            elif is_likely_person_name(tail):
                ent_id = f"e{id_counter}"
                id_counter += 1
                entities.append({"id": ent_id, "type": "Name", "text": tail})
                name_text_to_id[tail] = ent_id
                tail_id = ent_id
        if head_id and tail_id:
            relations.append({"type": "PAN_Of", "head": head_id, "tail": tail_id})

    return entities, relations


def finalize_and_save(aggregated_entities: List[Dict], aggregated_relations: List[Dict], out_dir: str) -> None:
    # Deduplicate entities by (type, text)
    seen = set()
    dedup_entities = []
    id_map = {}
    next_id = 1
    for e in aggregated_entities:
        key = (e["type"], e["text"])
        if key in seen:
            continue
        seen.add(key)
        new_id = f"E{next_id}"
        next_id += 1
        id_map[e["id"]] = new_id
        e_out = {"id": new_id, "type": e["type"], "text": e["text"]}
        dedup_entities.append(e_out)

    # Remap relations to dedup entity IDs; drop invalid
    valid_entity_ids = {e["id"] for e in dedup_entities}
    remapped_relations = []
    for r in aggregated_relations:
        head_old = r["head"]
        tail_old = r["tail"]
        head_new = id_map.get(head_old)
        tail_new = id_map.get(tail_old)
        if head_new in valid_entity_ids and tail_new in valid_entity_ids:
            remapped_relations.append({"type": r["type"], "head": head_new, "tail": tail_new})

    # Save CSVs
    os.makedirs(out_dir, exist_ok=True)
    entities_csv = os.path.join(out_dir, "entities.csv")
    relations_csv = os.path.join(out_dir, "relations.csv")

    pd.DataFrame(dedup_entities).to_csv(entities_csv, index=False)
    pd.DataFrame(remapped_relations).to_csv(relations_csv, index=False)

    # Also save a combined CSV per the example mapping PAN_Of
    pan_of_rows = []
    id_to_text = {e["id"]: e["text"] for e in dedup_entities}
    id_to_type = {e["id"]: e["type"] for e in dedup_entities}
    for r in remapped_relations:
        if r["type"] != "PAN_Of":
            continue
        pan_text = id_to_text.get(r["head"], "") if id_to_type.get(r["head"]) == "PAN" else ""
        name_text = id_to_text.get(r["tail"], "") if id_to_type.get(r["tail"]) == "Name" else ""
        if pan_text and name_text:
            pan_of_rows.append({"PAN": pan_text, "Relation": "PAN_Of", "Person": name_text})

    combined_csv = os.path.join(out_dir, "pan_of.csv")
    pd.DataFrame(pan_of_rows).to_csv(combined_csv, index=False)


def extract(pdf_path: str, model_repo: Optional[str], model_file: Optional[str], out_dir: str, preset: str, mode: str) -> None:
    pages = load_pdf_text(pdf_path)
    all_text = "\n\n".join(text for _, text in pages)
    pan_list = find_pans(all_text)

    # Resolve model to use
    if not model_repo or not model_file:
        preset_def = MODEL_PRESETS.get(preset, MODEL_PRESETS["qwen1_5b"])
        model_repo = preset_def["repo"]
        model_file = preset_def["file"]

    model_path = ensure_model(model_repo, model_file)
    llm = get_llm(model_path)

    aggregated_entities: List[Dict] = []
    aggregated_relations: List[Dict] = []

    if mode == "auto":
        mode = "pan_context" if pan_list else "chunks"

    if mode == "pan_context":
        pan_contexts = build_pan_contexts(pages, pan_list)
        for pan in tqdm(pan_list, desc="PAN linking"):
            contexts = pan_contexts.get(pan, [])
            if not contexts:
                entities, relations = reconcile([pan], {"entities": [], "relations": []})
                aggregated_entities.extend(entities)
                aggregated_relations.extend(relations)
                continue
            found_relation = False
            # Try up to first 2 contexts for this PAN
            for _, context in contexts[:2]:
                llm_json = llm_extract_for_pan(llm, pan, context)
                entities, relations = reconcile([pan], llm_json)
                aggregated_entities.extend(entities)
                aggregated_relations.extend(relations)
                if relations:
                    found_relation = True
                    break
            if not found_relation:
                # Heuristic fallback: single clear name in context
                context = contexts[0][1]
                names = heuristic_names_from_context(context)
                if len(names) == 1 and is_likely_person_name(names[0]):
                    # create minimal JSON to feed reconcile for consistent ID construction
                    llm_json = {"entities": [{"type": "Name", "text": names[0]}],
                                "relations": [{"type": "PAN_Of", "head": pan, "tail": names[0]}]}
                    entities, relations = reconcile([pan], llm_json)
                    aggregated_entities.extend(entities)
                    aggregated_relations.extend(relations)
    else:
        # Fallback: general chunking
        chunks = chunk_text(pages)
        for _, chunk in tqdm(chunks, desc="LLM extracting"):
            llm_json = llm_extract_chunk(llm, chunk)
            entities, relations = reconcile(pan_list, llm_json)
            aggregated_entities.extend(entities)
            aggregated_relations.extend(relations)

    finalize_and_save(aggregated_entities, aggregated_relations, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract entities (Organisation, Name, PAN) and PAN_Of relations from PDF")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--model-repo", type=str, default=None, help="HF repo id for GGUF (optional)")
    parser.add_argument("--model-file", type=str, default=None, help="GGUF filename (optional)")
    parser.add_argument("--preset", type=str, default="qwen1_5b", choices=list(MODEL_PRESETS.keys()), help="Model preset")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "pan_context", "chunks"], help="Extraction mode")
    args = parser.parse_args()

    extract(args.pdf, args.model_repo, args.model_file, args.out, args.preset, args.mode)


if __name__ == "__main__":
    main() 