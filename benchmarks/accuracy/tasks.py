"""Datasets, prompts and metrics of the accuracy experiments."""

import io
import json
import random
import re
import zipfile
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rouge_score import rouge_scorer


@dataclass
class Example:
    prompt: str
    reference: object


@dataclass
class Task:
    name: str
    metric: str
    max_new_tokens: int
    examples: list[Example]
    score: Callable[[str, object], float]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\- ]", "", text.lower()).strip()


def edit_similarity(a: str, b: str) -> float:
    """1 - normalized Levenshtein distance."""
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def imdb(limit: int, seed: int) -> Task:
    data = load_dataset("adrienheymans/imdb-movie-genres", split="test").shuffle(seed=seed).select(range(limit))
    genres = sorted(set(load_dataset("adrienheymans/imdb-movie-genres", split="test")["genre"]))
    options = ", ".join(genres)
    examples = [
        Example(
            f"Classify the genre of the movie described below. Choose one of: {options}.\n"
            f"Answer with the genre only.\n\nTitle: {row['title']}\nPlot: {row['text']}",
            row["genre"],
        )
        for row in data
    ]

    def score(output: str, reference: str) -> float:
        words = _normalize(output).split()
        return float(bool(words) and words[0] == _normalize(reference))

    return Task("imdb", "exact match", 8, examples, score)


def arxiv(limit: int, seed: int, max_chars: int = 40000) -> Task:
    path = hf_hub_download("ccdv/arxiv-summarization", "section/test-00000-of-00001.parquet", repo_type="dataset")
    data = load_dataset("parquet", data_files=path, split="train").shuffle(seed=seed).select(range(limit))
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    examples = [
        Example(f"Write the abstract of the following paper.\n\n{row['article'][:max_chars]}", row["abstract"])
        for row in data
    ]
    return Task("arxiv", "ROUGE-1", 256, examples, lambda out, ref: scorer.score(ref, out)["rouge1"].fmeasure)


def cocktail(limit: int, seed: int, subset: str = "nq", passages: int = 100) -> Task:
    archive = zipfile.ZipFile(hf_hub_download(f"IR-Cocktail/{subset}", f"{subset}.zip", repo_type="dataset"))

    def read_jsonl(name):
        with archive.open(name) as handle:
            return [json.loads(line) for line in io.TextIOWrapper(handle, encoding="utf-8")]

    corpus_files = sorted(n for n in archive.namelist() if n.startswith("corpus/") and n.endswith(".jsonl"))
    corpora = [{doc["_id"]: doc for doc in read_jsonl(name)} for name in corpus_files]
    queries = {q["_id"]: q["text"] for q in read_jsonl("queries.jsonl")}
    relevant: dict[str, set[str]] = {}
    with archive.open("qrels/test.tsv") as handle:
        for line in list(io.TextIOWrapper(handle, encoding="utf-8"))[1:]:
            query_id, doc_id, grade = line.split("\t")
            if int(grade) > 0:
                relevant.setdefault(query_id, set()).add(doc_id)

    rng = random.Random(seed)
    doc_ids = sorted(corpora[0])
    examples = []
    for query_id in rng.sample(sorted(relevant), limit):
        gold = rng.choice(sorted(relevant[query_id]))
        chosen = [gold] + rng.sample([d for d in doc_ids if d not in relevant[query_id]], passages - 1)
        rng.shuffle(chosen)
        lines = []
        for doc_id in chosen:
            doc = rng.choice(corpora)[doc_id]
            lines.append(f"[{doc_id}] {doc.get('title', '')}: {doc['text']}")
        prompt = (
            "Below is a list of passages, each starting with its identifier in brackets.\n\n"
            + "\n".join(lines)
            + f"\n\nQuestion: {queries[query_id]}\nWhich passage answers the question? Reply with its identifier only."
        )
        examples.append(Example(prompt, gold))

    def score(output: str, reference: str) -> float:
        found = re.findall(r"doc\d+", output)
        return float(bool(found) and found[0] == reference)

    return Task("cocktail", "exact match", 12, examples, score)


def humaneval(limit: int, seed: int) -> Task:
    data = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=seed).select(range(min(limit, 164)))
    examples = [
        Example(f"Complete the following Python function. Return only the code of the function body.\n\n{row['prompt']}",
                row["canonical_solution"])
        for row in data
    ]

    def score(output: str, reference: str) -> float:
        code = re.sub(r"```(?:python)?", "", output).strip()
        return edit_similarity(code, reference.strip())

    return Task("humaneval", "edit similarity", 256, examples, score)


def gsm8k(limit: int, seed: int) -> Task:
    data = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(limit))
    examples = [
        Example(f"{row['question']}\nSolve step by step. End with a line of the form 'Answer: <number>'.",
                row["answer"].split("####")[-1].strip().replace(",", ""))
        for row in data
    ]

    def score(output: str, reference: str) -> float:
        found = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", output) or re.findall(r"(-?[\d,]*\.?\d+)", output)
        try:
            return float(bool(found) and abs(float(found[-1].replace(",", "")) - float(reference)) < 1e-4)
        except ValueError:
            return 0.0

    return Task("gsm8k", "exact match", 512, examples, score)


TASKS = {"imdb": imdb, "arxiv": arxiv, "cocktail": cocktail, "humaneval": humaneval, "gsm8k": gsm8k}
