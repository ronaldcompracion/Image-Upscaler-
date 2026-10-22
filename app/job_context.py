from __future__ import annotations

from contextvars import ContextVar, Token
import logging

_job_id: ContextVar[str] = ContextVar("job_id", default="-")


def set_job_id(job_id: str) -> Token[str]:
    return _job_id.set(job_id)


def reset_job_id(token: Token[str]) -> None:
    _job_id.reset(token)


class JobContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()
        return True
