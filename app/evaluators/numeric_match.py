import math
import re

from app.models import MetricResult

from .base import Evaluator

# A number with optional sign, thousands separators, decimals and a
# denominator. The sign only counts when it does not follow a word or digit,
# so "2-1" yields 2 and 1, not -1.
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>-)?"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:/(?P<denominator>\d+(?:\.\d+)?))?"
)

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_NUMBER_WORD = re.compile(r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)

RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9


def extract_last_number(text: str) -> float | None:
    """Return the last number in the text, or None if there is none.

    Number words ("four") are only used when the text has no digits, so
    prose such as "one of the answers is 4" still yields 4.
    """

    matches = list(_NUMBER.finditer(text))

    if matches:
        last = matches[-1]
        value = float(last.group("value").replace(",", ""))
        denominator = last.group("denominator")

        if denominator is not None:
            divisor = float(denominator)

            if divisor == 0:
                return None

            value /= divisor

        return -value if last.group("sign") else value

    words = _NUMBER_WORD.findall(text)

    if words:
        return float(_NUMBER_WORDS[words[-1].lower()])

    return None


class NumericMatchEvaluator(Evaluator):
    """Compares the expected number with the last number in the answer.

    Using the last number follows common math benchmarks: in "2 + 2 = 4"
    the answer is 4. Numbers are compared with a small tolerance, so "4"
    matches "4.0".
    """

    def validate_expected_output(self, expected_output: str) -> None:
        if extract_last_number(expected_output) is None:
            raise ValueError("expected output does not contain a number")

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        expected = extract_last_number(expected_output)
        generated = extract_last_number(generated_output)

        if expected is None:
            raise ValueError("expected output does not contain a number")

        match = generated is not None and math.isclose(
            expected,
            generated,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )

        return MetricResult(
            metric_name="numeric_match",
            score=1.0 if match else 0.0,
            passed=match,
        )
