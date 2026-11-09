from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

import main
from api import routes
from ingestion.document_processor import process_document
from retrieval.vector_store import vector_store_manager as store
from utils.config import settings

client = TestClient(main.app)


def process(path, ext=".txt"):
    return process_document(str(path), ext, path.name, f"general/{path.name}", "general")


def login():
    token = client.post("/api/auth/login", json={"username": "staff", "password": "s3"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def make_table_pdf(path, rows):
    style = getSampleStyleSheet()["BodyText"]
    table = Table(rows, style=TableStyle([("GRID", (0, 0), (-1, -1), 1, "black")]))
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [Paragraph("REVENUE", style), Spacer(1, 10), table, Spacer(1, 10), Paragraph("Closing remark text.", style)]
    )


def test_every_piece_is_labelled_with_its_section(tmp_path):
    first = " ".join(f"First project detail number {i}." for i in range(60))
    second = "Meeting Agent is another project."
    path = tmp_path / "cv.txt"
    path.write_text(f"PROJECTS\n{first}\n{second}\n\nSKILLS\nPython")
    chunks = process(path)
    meeting = next(c for c in chunks if "Meeting Agent" in c.page_content)
    assert len([c for c in chunks if c.metadata["section"] == "PROJECTS"]) > 1
    assert meeting.page_content.startswith("Section: PROJECTS\n")
    assert chunks[-1].page_content == "Section: SKILLS\nPython"


def test_sentences_and_lists_are_not_headings(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("AWS, GCP, SQL\nA normal sentence.\nAPI | REST")
    assert {c.metadata["section"] for c in process(path)} == {""}


def test_pdf_table_keeps_rows_and_columns(tmp_path):
    path = tmp_path / "t.pdf"
    make_table_pdf(path, [["Region", "Q1", "Q2"], ["West", "10", "25"], ["East", "30", "40"]])
    chunks = process(path, ".pdf")
    table = next(c.page_content for c in chunks if "| West |" in c.page_content)
    assert table == "Section: REVENUE\n| Region | Q1 | Q2 |\n| --- | --- | --- |\n| West | 10 | 25 |\n| East | 30 | 40 |"
    assert any("Closing remark text." in c.page_content for c in chunks)


def test_long_table_repeats_header(tmp_path):
    path = tmp_path / "long.pdf"
    make_table_pdf(path, [["Name", "Value"]] + [[f"Item {i}", "Some longer value text"] for i in range(40)])
    pieces = [c.page_content for c in process(path, ".pdf") if "| Item" in c.page_content]
    assert len(pieces) > 1
    assert all("| Name | Value |" in piece for piece in pieces)
    assert all(len(piece) <= settings.CHUNK_SIZE + 50 for piece in pieces)


def test_search_brings_whole_section_in_reading_order(tmp_path):
    path = tmp_path / "sectioned.txt"
    filler = " ".join(f"Zebra habitat fact {i}." for i in range(80))
    path.write_text(f"ANIMALS\n{filler}\nThe platypus lays eggs.\n\nOTHER\nCars have wheels.")
    chunks = process(path)
    store.replace_document(chunks[0].metadata["doc_path"], chunks)
    found = [
        d for d in store.search_with_context("platypus eggs", top_k=1, role="general")
        if d.metadata["source"] == path.name
    ]
    indexes = [d.metadata["chunk_index"] for d in found]
    assert len(found) == len([c for c in chunks if c.metadata["section"] == "ANIMALS"]) > 1
    assert indexes == sorted(indexes)


def test_follow_up_receives_history_and_skips_cache(monkeypatch):
    calls = []

    def answer(query, role="general", history=None):
        calls.append(history)
        return {"answer": "ok", "sources": ["a.txt"]}

    monkeypatch.setattr(routes, "get_answer", answer)
    turns = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    body = {"query": "And the second one?", "history": turns}
    for _ in range(2):
        response = client.post("/api/query", json=body, headers=login())
        assert response.json()["cached"] is False
    assert len(calls) == 2
    assert [t["content"] for t in calls[0]] == [f"message {i}" for i in range(4, 10)]
