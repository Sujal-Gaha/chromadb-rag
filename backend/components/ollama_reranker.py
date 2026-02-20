from typing import Any
from haystack import Document, component
from haystack_integrations.components.generators.ollama import OllamaGenerator


@component
class OllamaReranker:
    def __init__(
        self,
        model: str,
        url: str,
        timeout: int = 120,
        top_k: int = 5,
    ):
        self.top_k = top_k

        self.generator = OllamaGenerator(
            model=model,
            url=url,
            timeout=timeout,
            generation_kwargs={
                "temperature": 0.0,
                "num_predict": 50,
                "top_p": 0.9,
            },
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], query: str) -> dict[str, Any]:
        """
        Rerank documents using Ollama LLM.
        """

        if not documents:
            return {"documents": []}

        scored_docs = []

        for doc in documents:
            prompt = f"""
            You are a document relevance scorer.

            Query: {query}

            Document: {doc.content}

            Give a relevance score between 0 and 100.
            Return ONLY the number.
            """

            score = 0.0

            try:
                result = self.generator.run(prompt=prompt)
                reply = result["replies"][0].strip()

                # Extract numeric score safely
                score = float("".join(c for c in reply if c.isdigit() or c == "."))

            except Exception:
                score = 0.0

            doc.score = score
            scored_docs.append(doc)

        # Sort by score descending
        ranked = sorted(scored_docs, key=lambda d: d.score, reverse=True)

        return {"documents": ranked[: self.top_k]}
