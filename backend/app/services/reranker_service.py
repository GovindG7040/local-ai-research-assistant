from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):

        print("Loading reranker model...")

        self.model = CrossEncoder(model_name)

        print("Reranker model loaded successfully.")

    # =====================================================
    # RERANK DOCUMENTS
    # =====================================================

    def rerank(
        self,
        question: str,
        documents: list,
        top_k: int = 5,
    ):
        """
        Returns the top_k most relevant documents.
        """

        if not documents:

            return []

        # ------------------------------------------
        # Create Question-Document Pairs
        # ------------------------------------------

        sentence_pairs = [

            (
                question,
                document.page_content,
            )

            for document in documents

        ]

        # ------------------------------------------
        # Predict Relevance Scores
        # ------------------------------------------

        scores = self.model.predict(sentence_pairs)

        print("\n========== RERANKER ==========")

        print(f"Retrieved Documents : {len(documents)}")

        # ------------------------------------------
        # Combine Scores with Documents
        # ------------------------------------------

        scored_documents = list(

            zip(
                documents,
                scores,
            )

        )

        # ------------------------------------------
        # Sort by Score (Descending)
        # ------------------------------------------

        scored_documents.sort(

            key=lambda item: item[1],

            reverse=True,

        )

        print("\nTop Ranked Documents:")

        for rank, (document, score) in enumerate(scored_documents[:top_k], start=1):

            print(f"{rank}. {document.metadata.get('source')}")

            print(f"   Score : {score:.4f}")

        # ------------------------------------------
        # Return Top Documents
        # ------------------------------------------

        reranked_documents = [

            document

            for document, score in scored_documents[:top_k]

        ]

        return reranked_documents