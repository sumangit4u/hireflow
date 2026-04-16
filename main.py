"""HireFlow - Simple Candidate Search CLI"""

from core.vector_store import VectorStore


def search_candidates(query: str, top_k: int = 5):
    vs = VectorStore()
    if not vs.initialize():
        print("Failed to initialize vector store")
        return []

    results = vs.search_resumes(query, top_k=top_k)
    return results


def main():
    """Simple candidate search interface"""
    print("HireFlow - Candidate Search Engine")

    while True:
        print("\n1. Search Candidates  2. Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            query = input("Enter search query (e.g. 'Senior Accountant with QuickBooks'): ").strip()
            if not query:
                print("Query cannot be empty")
                continue

            candidates = search_candidates(query, top_k=5)

            if candidates:
                print(f"\nFound {len(candidates)} candidates:")
                for i, c in enumerate(candidates[:5], 1):
                    name = c.get('metadata', {}).get('name', 'Unknown')
                    candidate_id = c.get('metadata', {}).get('candidate_id', '')
                    score = c.get('score', 0)
                    print(f"   {i}. {name} (ID: {candidate_id}, Score: {score:.3f})")
            else:
                print("No candidates found")

        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
