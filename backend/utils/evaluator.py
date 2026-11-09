from retrieval.vector_store import vector_store_manager


def evaluate_retrieval(queries, ground_truths, top_k=5):
    """
    File-level Precision, Recall and F1 for retrieval, across all documents.
    queries: List[str]
    ground_truths: List[List[str]], the file names that should be retrieved for each query
    """
    if not queries:
        return {}

    total_precision = 0.0
    total_recall = 0.0

    for query, expected in zip(queries, ground_truths):
        docs = vector_store_manager.search(query, top_k=top_k, role="Admin")
        retrieved = {doc.metadata.get("source", "") for doc in docs}
        expected = set(expected)
        hits = len(retrieved & expected)
        total_precision += hits / len(retrieved) if retrieved else 0
        total_recall += hits / len(expected) if expected else 0

    precision = total_precision / len(queries)
    recall = total_recall / len(queries)
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score}
