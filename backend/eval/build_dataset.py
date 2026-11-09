"""Build the 250-question benchmark: SQuAD v2, HotpotQA and generated PDF tables."""
import hashlib
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SEED = 42
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
)
OUT = Path(__file__).parent / "data"
DOCS = OUT / "docs"


def download(url, name):
    """Fetch a file into the data folder once."""
    target = OUT / name
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "rag-eval"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    return target


def write_doc(name, text):
    (DOCS / name).write_text(text, encoding="utf-8")
    return name


def build_squad(rng):
    # Spread across articles so the corpus covers many topics.
    source = json.loads(download(SQUAD_URL, "squad_dev.json").read_text(encoding="utf-8"))
    rows = []
    for article in source["data"][::2]:
        for paragraph in article["paragraphs"][:8]:
            for qa in paragraph["qas"]:
                rows.append(
                    {"context": paragraph["context"], "question": qa["question"],
                     "answers": {"text": [a["text"] for a in qa["answers"]]}}
                )

    files = {}
    for row in rows:
        if row["context"] not in files:
            files[row["context"]] = write_doc(f"squad_{len(files):04}.txt", row["context"])

    rng.shuffle(rows)
    answerable = [r for r in rows if r["answers"]["text"]][:100]
    impossible = [r for r in rows if not r["answers"]["text"]][:50]
    questions = []
    for r in answerable:
        questions.append(
            {"category": "fact", "question": r["question"], "gold": sorted(set(r["answers"]["text"])),
             "gold_sources": [files[r["context"]]], "evidence": True}
        )
    for r in impossible:
        questions.append(
            {"category": "unanswerable", "question": r["question"], "gold": [],
             "gold_sources": [files[r["context"]]], "evidence": False}
        )
    return questions


def build_hotpot(rng):
    table = pq.read_table(download(HOTPOT_URL, "hotpot_validation.parquet"))
    rows = table.slice(0, 200).to_pylist()

    files = {}
    for row in rows:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in files:
                name = f"wiki_{hashlib.sha1(title.encode()).hexdigest()[:10]}.txt"
                files[title] = write_doc(name, f"{title}\n{''.join(sentences)}")

    usable = [r for r in rows if r["answer"].lower() not in ("yes", "no")]
    rng.shuffle(usable)
    return [
        {"category": "multi_hop", "question": r["question"], "gold": [r["answer"]],
         "gold_sources": sorted({files[t] for t in r["supporting_facts"]["title"]}), "evidence": False}
        for r in usable[:50]
    ]


THEMES = [
    ("warehouse_inventory.pdf", "WAREHOUSE INVENTORY", "Warehouse", "units in stock", "monthly shipments",
     ["Aldgate", "Birchwood", "Cresthaven", "Dunmere", "Elmfield", "Fernley", "Glenrock", "Harlow", "Ironbridge", "Juniper"]),
    ("regional_sales.pdf", "REGIONAL SALES", "Territory", "Q1 sales", "Q2 sales",
     ["Kestrel", "Larkspur", "Meridian", "Northgate", "Oakhurst", "Pinecrest", "Quarry", "Ridgeway", "Stonebrook", "Tamarack"]),
    ("employee_headcount.pdf", "EMPLOYEE HEADCOUNT", "Department", "full-time staff", "contractors",
     ["Actuarial", "Billing", "Compliance", "Dispatch", "Engineering", "Facilities", "Graphics", "Helpdesk", "Inspection", "Janitorial"]),
    ("vendor_contracts.pdf", "VENDOR CONTRACTS", "Vendor", "contract value", "contract months",
     ["Acorn Ltd", "Beacon Co", "Cobalt Inc", "Delta Works", "Ember Group", "Falcon Tech", "Granite Co", "Horizon Ltd", "Indigo Inc", "Jasper Group"]),
    ("project_budgets.pdf", "PROJECT BUDGETS", "Project", "approved budget", "actual spend",
     ["Atlas", "Borealis", "Cascade", "Dynamo", "Eclipse", "Fusion", "Galaxy", "Helios", "Ion", "Jupiter"]),
]


def build_tables(rng):
    style = getSampleStyleSheet()["BodyText"]
    questions = []
    for filename, heading, key, col_a, col_b, names in THEMES:
        values_a = rng.sample(range(100, 999), len(names))
        values_b = rng.sample(range(100, 999), len(names))
        managers = rng.sample(
            ["Priya", "Arjun", "Meera", "Kabir", "Nisha", "Rohan", "Sara", "Tara", "Vikram", "Zoya"], len(names)
        )
        data = list(zip(names, values_a, values_b, managers))
        rows = [[key, col_a, col_b, "Manager"]] + [[n, str(a), str(b), m] for n, a, b, m in data]
        table = Table(rows, style=TableStyle([("GRID", (0, 0), (-1, -1), 1, "black")]))
        SimpleDocTemplate(str(DOCS / filename), pagesize=letter).build(
            [Paragraph(heading, style), Spacer(1, 12), table]
        )

        topic = heading.lower()
        picks = rng.sample(data, 6)
        prompts = [
            (f"In the {topic} table, what is the {col_a} for {picks[0][0]}?", str(picks[0][1])),
            (f"In the {topic} table, what is the {col_b} for {picks[1][0]}?", str(picks[1][2])),
            (f"In the {topic} table, what is the {col_a} for {picks[2][0]}?", str(picks[2][1])),
            (f"In the {topic} table, what is the {col_b} for {picks[3][0]}?", str(picks[3][2])),
            (f"In the {topic} table, who is the manager of {picks[4][0]}?", picks[4][3]),
            (f"In the {topic} table, who is the manager of {picks[5][0]}?", picks[5][3]),
            (f"In the {topic} table, which {key.lower()} has the highest {col_a}?", max(data, key=lambda d: d[1])[0]),
            (f"In the {topic} table, which {key.lower()} has the lowest {col_a}?", min(data, key=lambda d: d[1])[0]),
            (f"In the {topic} table, which {key.lower()} has the highest {col_b}?", max(data, key=lambda d: d[2])[0]),
            (f"In the {topic} table, what is the difference between {col_a} and {col_b} for {picks[0][0]}? "
             "Give the absolute value.", str(abs(picks[0][1] - picks[0][2]))),
        ]
        for question, gold in prompts:
            questions.append(
                {"category": "table", "question": question, "gold": [gold], "gold_sources": [filename], "evidence": False}
            )
    return questions


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    questions = build_squad(rng) + build_hotpot(rng) + build_tables(rng)
    for number, question in enumerate(questions):
        question["id"] = number
    with open(OUT / "questions.jsonl", "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question) + "\n")
    counts = {}
    for question in questions:
        counts[question["category"]] = counts.get(question["category"], 0) + 1
    print(counts, "documents:", len(list(DOCS.iterdir())))


if __name__ == "__main__":
    main()
