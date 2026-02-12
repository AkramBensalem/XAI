import json
from typing import Dict, Any, List, Union
import re
from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

class NLTKPromptEngineer:
    """A class for managing prompt engineering with NLTK analysis."""

    def __init__(self, logging):
        """
        Initialize the NLTKPromptEngineer class.
        """
        self.logging = logging
        self.history: List[Dict[str, Any]] = []

        # Initialize NLTK analyzers
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

        self.logging.info("NLTK Prompt Engineer initialized successfully")

    def create_structured_prompt(
        self,
        task: str,
        context: str = "",
        constraints: List[str] = None,
        examples: List[Dict[str, str]] = None
    ) -> str:
        """
        Create a structured prompt following best practices.

        Args:
            task (str): Main task description
            context (str): Additional context for the task
            constraints (List[str]): List of constraints to apply
            examples (List[Dict[str, str]]): List of example input/output pairs

        Returns:
            str: Formatted text prompt
        """
        constraints = constraints or []
        examples = examples or []

        # Build prompt content
        prompt_text = "# Task\n" + task

        if context:
            prompt_text += "\n\n# Context\n" + context

        if constraints:
            prompt_text += "\n\n# Constraints:\n"
            prompt_text += "\n".join(f"- {c}" for c in constraints)

        # Add examples
        if examples:
            prompt_text += "\n\n# Examples:\n"
            for example in examples:
                if "input" in example:
                    prompt_text += f"\nInput: {example['input']}\n"
                if "output" in example:
                    prompt_text += f"Output: {example['output']}\n"

        return prompt_text

    def analyze_text(
        self,
        text: str,
        analyze_sentiment: bool = True,
        analyze_bias: bool = True,
        analyze_complexity: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze text using NLTK for various metrics.

        Args:
            text (str): Text to analyze
            analyze_sentiment (bool): Whether to analyze sentiment
            analyze_bias (bool): Whether to analyze bias
            analyze_complexity (bool): Whether to analyze complexity

        Returns:
            Dict[str, Any]: Analysis metrics
        """
        results = {}

        # Basic text stats
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        results["word_count"] = len(words)
        results["sentence_count"] = len(sentences)
        results["avg_words_per_sentence"] = len(words) / len(sentences) if sentences else 0

        # Sentiment analysis
        if analyze_sentiment:
            sentiment = self.sia.polarity_scores(text)
            results["sentiment"] = sentiment

        # Bias analysis
        if analyze_bias:
            bias_scores = self._evaluate_bias(text)
            results["bias"] = bias_scores

        # Text complexity
        if analyze_complexity:
            # Simple readability metrics
            long_words = [w for w in words if len(w) > 6]
            results["complexity"] = {
                "long_word_ratio": len(long_words) / len(words) if words else 0,
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            }

        return results

    # Pre-compiled regex patterns for bias evaluation
    _bias_indicators = {
        'gender_bias': {
            'patterns': [
                re.compile(r'\b(he|his|him|gentleman|man|men)\b(?!.*\b(she|her|hers|lady|woman|women)\b)', re.IGNORECASE),
                re.compile(r'\b(she|her|hers|lady|woman|women)\b(?!.*\b(he|his|him|gentleman|man|men)\b)', re.IGNORECASE),
                re.compile(r'\b(businessman|businesswoman|chairman|chairwoman|spokesman|spokeswoman)\b', re.IGNORECASE)
            ],
            'weight': 0.3
        },
        'racial_bias': {
            'patterns': [
                re.compile(r'\b(normal|standard|regular|typical|default)(?=\s+(person|people|individual|community))\b', re.IGNORECASE),
                re.compile(r'\b(ethnic|minority|diverse)(?=\s+only\b)', re.IGNORECASE),
            ],
            'weight': 0.3
        },
        'age_bias': {
            'patterns': [
                re.compile(r'\b(young|old|elderly|senior)(?=\s+people\b)', re.IGNORECASE),
                re.compile(r'\b(millennials|boomers|gen\s+[xyz])\b\s+(?=\b(are|always|never|typically)\b)', re.IGNORECASE),
            ],
            'weight': 0.2
        },
        'socioeconomic_bias': {
            'patterns': [
                re.compile(r'\b(poor|rich|wealthy|low-income|high-income)(?=\s+people\b)', re.IGNORECASE),
                re.compile(r'\b(educated|uneducated|privileged|underprivileged)\b', re.IGNORECASE),
            ],
            'weight': 0.2
        }
    }

    def _evaluate_bias(self, text: str) -> Dict[str, float]:
        """
        Evaluate text for various types of bias using NLTK and regex.

        Args:
            text (str): Text to evaluate

        Returns:
            Dict[str, float]: Bias scores for different bias types
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        bias_scores = {}
        overall_bias = 0.0

        for bias_type, config in self._bias_indicators.items():
            type_score = 0
            matches = []

            for pattern in config['patterns']:
                found_matches = pattern.findall(text)
                matches.extend(found_matches)
                if found_matches:
                    type_score += len(found_matches) * 0.1

            bias_scores[bias_type] = min(1.0, type_score)
            overall_bias += bias_scores[bias_type] * config['weight']

            # Store matched phrases for explanation
            bias_scores[f"{bias_type}_matches"] = matches

        bias_scores["overall"] = min(1.0, overall_bias)
        return bias_scores

    def evaluate_text(
        self,
        text: str,
        criteria: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate the quality of text based on given criteria.

        Args:
            text (str): Text to evaluate
            criteria (List[str]): List of evaluation criteria

        Returns:
            Dict[str, float]: Evaluation scores
        """
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        scores = {}

        print(f"Evaluating text on {len(criteria)} criteria...")

        for i, criterion in enumerate(criteria):
            if criterion == "bias":
                bias_results = self._evaluate_bias(text)
                scores[criterion] = bias_results["overall"]
                # Add specific bias types
                for bias_type in self._bias_indicators.keys():
                    scores[f"bias_{bias_type}"] = bias_results[bias_type]
            elif criterion == "sentiment":
                sentiment = self.sia.polarity_scores(text)
                scores["sentiment_positive"] = sentiment["pos"]
                scores["sentiment_negative"] = sentiment["neg"]
                scores["sentiment_neutral"] = sentiment["neu"]
                scores["sentiment_compound"] = sentiment["compound"]
            elif criterion == "clarity":
                # Measure clarity based on sentence length, word complexity
                words = word_tokenize(text)
                sentences = sent_tokenize(text)
                avg_sentence_length = len(words) / len(sentences) if sentences else 0
                complex_words = [w for w in words if len(w) > 6 and w.lower() not in self.stop_words]
                scores["clarity"] = 1.0 - min(1.0, (len(complex_words) / len(words) * 1.5 +
                                           (avg_sentence_length / 25.0)))
            elif criterion == "engagement":
                # Measure engagement based on question marks, imperative verbs, etc.
                question_count = text.count("?")
                exclamation_count = text.count("!")
                second_person_count = len(re.findall(r'\byou\b|\byour\b', text, re.IGNORECASE))
                engagement_score = min(1.0, (question_count * 0.2 + exclamation_count * 0.1 +
                                          second_person_count * 0.05))
                scores["engagement"] = engagement_score
            else:
                # Default to a neutral score for unknown criteria
                scores[criterion] = 0.5

            print(f"Evaluated {i+1}/{len(criteria)}: {criterion}")

        print("Evaluation complete!")
        return scores

    def save_history(self, filepath: Union[str, Path]) -> None:
        """
        Save interaction history to a JSON file.

        Args:
            filepath (Union[str, Path]): Path to save the history file
        """
        print(f"Saving history to {filepath}...")
        filepath = Path(filepath)
        with filepath.open('w') as f:
            json.dump(self.history, f, indent=2)
        print("History saved successfully!")