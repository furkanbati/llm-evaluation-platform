from app.models import EvaluationErrorCode


class EvaluationError(Exception):
    """Base class for failures that map to a public error category.

    The exception message holds internal details for logs. `public_message`
    is safe to show to API clients.
    """

    code = EvaluationErrorCode.INTERNAL_ERROR
    public_message = "An internal error occurred while running the evaluation."

    def __init__(
        self,
        detail: str = "",
        public_message: str | None = None,
    ) -> None:
        super().__init__(detail or self.public_message)

        if public_message is not None:
            self.public_message = public_message


class ModelUnavailableError(EvaluationError):
    code = EvaluationErrorCode.MODEL_UNAVAILABLE
    public_message = "Could not connect to the model provider."


class ModelTimeoutError(EvaluationError):
    code = EvaluationErrorCode.MODEL_TIMEOUT
    public_message = "The model provider did not respond in time."


class ModelNotFoundError(EvaluationError):
    code = EvaluationErrorCode.MODEL_NOT_FOUND

    def __init__(
        self,
        model_name: str,
        detail: str = "",
    ) -> None:
        super().__init__(
            detail,
            public_message=(
                f"Model '{model_name}' is not available on the model provider."
            ),
        )


class ModelProviderError(EvaluationError):
    code = EvaluationErrorCode.MODEL_PROVIDER_ERROR
    public_message = "The model provider returned an error."


class InvalidModelOutputError(EvaluationError, ValueError):
    code = EvaluationErrorCode.INVALID_MODEL_OUTPUT
    public_message = "The model returned an invalid output."


class InvalidJudgeResponseError(EvaluationError, ValueError):
    code = EvaluationErrorCode.INVALID_JUDGE_RESPONSE
    public_message = "The LLM judge returned a response that was not YES or NO."
