"""
Comprehensive NLP evaluation metrics for RAG
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util

# Try to import NLP metrics, but provide fallbacks
try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed. Install with: pip install nltk")

try:
    from bert_score import score as bert_score

    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert-score not installed. Install with: pip install bert-score")


@dataclass
class MetricScores:
    """Container for all evaluation metrics"""

    # Semantic similarity
    cosine_similarity: float
    semantic_similarity: float

    # Token-based metrics
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    meteor_score: Optional[float] = None

    # BERT-based metrics
    bert_precision: Optional[float] = None
    bert_recall: Optional[float] = None
    bert_f1: Optional[float] = None

    # Custom metrics
    exact_match: bool = False
    partial_match: float = 0.0
    keyword_overlap: float = 0.0


class NLPMetrics:
    """Comprehensive NLP metrics calculator"""

    def __init__(self, similarity_model_name: str = "all-MiniLM-L6-v2"):
        self.similarity_model = SentenceTransformer(similarity_model_name)

        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
            )

        if NLTK_AVAILABLE:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("wordnet", quiet=True)
            except:
                pass

    def calculate_all_metrics(self, reference: str, candidate: str) -> MetricScores:
        """Calculate all available metrics"""
        metrics = MetricScores(
            cosine_similarity=self._cosine_similarity(reference, candidate),
            semantic_similarity=self._semantic_similarity(reference, candidate),
            exact_match=self._exact_match(reference, candidate),
            partial_match=self._partial_match(reference, candidate),
            keyword_overlap=self._keyword_overlap(reference, candidate),
        )

        # Calculate BLEU if available
        if NLTK_AVAILABLE:
            metrics.bleu_score = self._bleu_score(reference, candidate)
            metrics.meteor_score = self._meteor_score(reference, candidate)

        # Calculate ROUGE if available
        if ROUGE_AVAILABLE:
            metrics.rouge_scores = self._rouge_scores(reference, candidate)

        # Calculate BERTScore if available
        if BERT_SCORE_AVAILABLE:
            bert_p, bert_r, bert_f = self._bert_score(reference, candidate)
            metrics.bert_precision = float(bert_p.mean())
            metrics.bert_recall = float(bert_r.mean())
            metrics.bert_f1 = float(bert_f.mean())

        return metrics

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between texts"""
        if not text1 or not text2:
            return 0.0

        vec1 = self.similarity_model.encode(text1, convert_to_tensor=True)
        vec2 = self.similarity_model.encode(text2, convert_to_tensor=True)
        return float(util.cos_sim(vec1, vec2).item())

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Alias for cosine similarity (same calculation)"""
        return self._cosine_similarity(text1, text2)

    def _bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not NLTK_AVAILABLE or not reference or not candidate:
            return 0.0

        try:
            ref_tokens = [word_tokenize(reference.lower())]
            cand_tokens = word_tokenize(candidate.lower())

            # Use smoothing for short texts
            smoothing = SmoothingFunction().method1
            return sentence_bleu(
                ref_tokens,
                cand_tokens,
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25),  # Default BLEU-4
            )
        except:
            return 0.0

    def _rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE or not reference or not candidate:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {k: v.fmeasure for k, v in scores.items()}
        except:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    def _meteor_score(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        if not NLTK_AVAILABLE or not reference or not candidate:
            return 0.0

        try:
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            return meteor_score([ref_tokens], cand_tokens)
        except:
            return 0.0

    def _bert_score(
        self, reference: str, candidate: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate BERTScore"""
        if not BERT_SCORE_AVAILABLE or not reference or not candidate:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

        try:
            P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
            return P.numpy(), R.numpy(), F1.numpy()
        except:
            return np.array([0.0]), np.array([0.0]), np.array([0.0])

    def _exact_match(self, reference: str, candidate: str) -> bool:
        """Check for exact string match (case-insensitive)"""
        if not reference or not candidate:
            return False
        return reference.strip().lower() == candidate.strip().lower()

    def _partial_match(self, reference: str, candidate: str) -> float:
        """Calculate partial match ratio"""
        if not reference or not candidate:
            return 0.0

        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())

        if not ref_words:
            return 0.0

        intersection = ref_words.intersection(cand_words)
        return len(intersection) / len(ref_words)

    def _keyword_overlap(self, reference: str, candidate: str) -> float:
        """Calculate keyword overlap using TF-IDF like approach"""
        if not reference or not candidate:
            return 0.0

        # Simple implementation - count common non-stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
        }

        ref_words = [w.lower() for w in reference.split() if w.lower() not in stopwords]
        cand_words = [
            w.lower() for w in candidate.split() if w.lower() not in stopwords
        ]

        if not ref_words:
            return 0.0

        common_words = set(ref_words).intersection(set(cand_words))
        return len(common_words) / len(ref_words)

    def get_metrics_summary(self, metrics: MetricScores) -> Dict[str, float]:
        """Get a summary dictionary of all metrics"""
        summary = {
            "cosine_similarity": metrics.cosine_similarity,
            "semantic_similarity": metrics.semantic_similarity,
            "exact_match": float(metrics.exact_match),
            "partial_match": metrics.partial_match,
            "keyword_overlap": metrics.keyword_overlap,
        }

        if metrics.bleu_score is not None:
            summary["bleu_score"] = metrics.bleu_score

        if metrics.meteor_score is not None:
            summary["meteor_score"] = metrics.meteor_score

        if metrics.rouge_scores:
            for rouge_type, score in metrics.rouge_scores.items():
                summary[f"{rouge_type}"] = score

        if metrics.bert_precision is not None:
            summary["bert_precision"] = metrics.bert_precision
            summary["bert_recall"] = metrics.bert_recall
            summary["bert_f1"] = metrics.bert_f1

        return summary


# Simple metrics calculator without dependencies
class SimpleMetrics:
    """Simple metrics calculator for when dependencies aren't available"""

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def calculate_basic_metrics(
        self, reference: str, candidate: str
    ) -> Dict[str, float]:
        """Calculate basic metrics without external dependencies"""
        if not reference or not candidate:
            return {
                "cosine_similarity": 0.0,
                "exact_match": 0.0,
                "partial_match": 0.0,
                "keyword_overlap": 0.0,
            }

        # Cosine similarity
        vec1 = self.model.encode(reference, convert_to_tensor=True)
        vec2 = self.model.encode(candidate, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(vec1, vec2).item())

        # Exact match
        exact_match = float(reference.strip().lower() == candidate.strip().lower())

        # Partial match
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        partial_match = len(ref_words.intersection(cand_words)) / max(len(ref_words), 1)

        # Keyword overlap (without stopwords)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
        ref_keywords = [w for w in reference.lower().split() if w not in stopwords]
        cand_keywords = [w for w in candidate.lower().split() if w not in stopwords]

        if not ref_keywords:
            keyword_overlap = 0.0
        else:
            common = set(ref_keywords).intersection(set(cand_keywords))
            keyword_overlap = len(common) / len(ref_keywords)

        return {
            "cosine_similarity": cosine_sim,
            "exact_match": exact_match,
            "partial_match": partial_match,
            "keyword_overlap": keyword_overlap,
        }
