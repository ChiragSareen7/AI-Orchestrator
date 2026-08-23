from __future__ import annotations  # allows modern type hint syntax on older Python

import os  # for reading environment variables
from contextlib import contextmanager  # for creating context managers using generator functions


_TRACEABLE = None
# module-level variable that stores the LangSmith 'traceable' decorator (or False if unavailable)
# None = haven't tried to load it yet; False = tried and it's not installed


def _load_traceable():
    # lazy-loads the LangSmith traceable decorator only when first needed
    # "lazy" means we don't import it at startup — only when actually called
    global _TRACEABLE  # tells Python we're modifying the module-level variable, not creating a local one

    if _TRACEABLE is not None:
        return _TRACEABLE  # already loaded (or already know it's unavailable) — return immediately

    try:
        from langsmith import traceable as t  # type: ignore
        # type: ignore tells the type checker to not complain about this optional import
        _TRACEABLE = t  # store the decorator function for future use
    except Exception:
        _TRACEABLE = False
        # langsmith is not installed or failed to import → set to False
        # False is falsy so the callers can check `if traceable:` safely

    return _TRACEABLE


def traceable_if_enabled(run_type: str, name: str):
    # creates a decorator that optionally wraps a function with LangSmith tracing
    # this is used as: @traceable_if_enabled(run_type="chain", name="execute-model-call")

    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    # check if LangSmith tracing is enabled in .env
    # LANGCHAIN_TRACING_V2=true → tracing_on = True
    # LANGCHAIN_TRACING_V2=false (or not set) → tracing_on = False

    traceable = _load_traceable()  # try to get the LangSmith decorator

    if tracing_on and traceable:
        # both conditions met: env var says ON and the library is available
        return traceable(run_type=run_type, name=name)
        # return the ACTUAL LangSmith decorator — it will send data to LangSmith

    def _noop(fn):
        return fn  # no-op decorator: just returns the function unchanged (does nothing)

    return _noop
    # if tracing is disabled OR langsmith not installed → return a do-nothing decorator
    # this means @traceable_if_enabled(...) has ZERO effect when tracing is off


@contextmanager
def trace_block(name: str):
    # a simple context manager for manually marking a block of code as a trace boundary
    # @contextmanager turns a generator function into a context manager
    # usage: with trace_block("my-step"): ... do stuff ...
    yield  # 'yield' is where the 'with' block's code runs
    # currently does nothing (lightweight boundary for code readability)
    # in a future version, this could send span data to a tracing system
