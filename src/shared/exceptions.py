"""Domain exception hierarchy for the medical AI assistant."""


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code or "APP_ERROR"
        super().__init__(self.message)


class ModelError(AppError):
    """LLM model invocation failure."""

    def __init__(self, message: str, model: str | None = None, status_code: int | None = None):
        self.model = model
        self.status_code = status_code
        super().__init__(message, code="MODEL_ERROR")


class RetrievalError(AppError):
    """Vector/KG retrieval failure."""

    def __init__(self, message: str, source: str | None = None):
        self.source = source or "unknown"
        super().__init__(message, code="RETRIEVAL_ERROR")


class KnowledgeBaseError(AppError):
    """Knowledge base ingestion/validation failure."""

    def __init__(self, message: str, doc_id: str | None = None):
        self.doc_id = doc_id
        super().__init__(message, code="KB_ERROR")


class EvalError(AppError):
    """Evaluation framework error."""

    def __init__(self, message: str, metric: str | None = None):
        self.metric = metric
        super().__init__(message, code="EVAL_ERROR")
