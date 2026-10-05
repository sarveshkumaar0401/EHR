import argparse

from main import answer_query, retrieve

COLLECTIONS = ("aci_bench_graph_ready", "aci_bench_clinical_notes")


def ask(question, args):
    print(f"\nCollection: {args.collection}")
    if args.retrieve_only:
        results = retrieve(
            question,
            patient_id=args.patient_id,
            symptom_filter=args.symptom,
            top_k=args.top_k,
            collection=args.collection,
        )
        if not results["contexts"]:
            print("No matching notes were found in the selected Qdrant collection.")
            return

        print(f"Retrieved {len(results['contexts'])} matching note chunk(s):")
        for index, context in enumerate(results["contexts"]):
            patient_id = results["patient_ids"][index] if index < len(results["patient_ids"]) else "N/A"
            doc_id = results["doc_ids"][index] if index < len(results["doc_ids"]) else "N/A"
            print(f"\n[{index + 1}] Patient: {patient_id} | Document: {doc_id}")
            print(context)
        return

    answer = answer_query(
        question,
        patient_id=args.patient_id,
        symptom_filter=args.symptom,
        top_k=args.top_k,
        show_context=True,
        collection=args.collection,
    )
    print("\n--- Answer ---")
    print(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve clinical-note evidence and answer questions from Qdrant."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask. Omit it to enter interactive mode.",
    )
    parser.add_argument(
        "--patient-id",
        help="Optional exact patient_id filter stored in Qdrant.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of context chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--collection",
        choices=COLLECTIONS,
        default="aci_bench_graph_ready",
        help="Qdrant collection to query (default: graph-ready collection).",
    )
    parser.add_argument(
        "--symptom",
        action="append",
        help="Optional canonical symptom filter; may be specified more than once.",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print matching Qdrant note chunks directly without generating an answer.",
    )
    args = parser.parse_args()

    if args.question:
        ask(args.question, args)
        return

    print(
        f"Querying Qdrant collection '{args.collection}'. "
        "Type 'exit' to quit."
    )
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(question, args)


if __name__ == "__main__":
    main()
