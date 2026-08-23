from __future__ import annotations  # allows modern type hint syntax on older Python

from typing import Any, Callable  # type hint utilities


_hooks: list[Callable[[str, dict[str, Any]], None]] = []
# module-level list that stores registered hook functions
# each hook is a callable (function) that accepts:
#   - event: str       (e.g. "semantic_eval")
#   - payload: dict    (the event data, e.g. scores and metadata)
# and returns nothing (None)
# starts empty — hooks are added by calling register_eval_hook()


def register_eval_hook(fn: Callable[[str, dict[str, Any]], None]) -> None:
    # registers a function to be called whenever an evaluation event fires
    # this is the "observer pattern" — interested parties register to receive events
    # example use: hook that logs eval results to a database or external monitoring tool
    _hooks.append(fn)
    # appends the function to our list of hooks


def emit_eval_event(event: str, payload: dict[str, Any]) -> None:
    # fires an event to ALL registered hooks
    # called by pipeline.py after each semantic evaluation completes
    for fn in _hooks:
        # loop through every registered hook function
        try:
            fn(event, payload)
            # call the hook with the event name and payload data
            # each hook can do whatever it wants: log, send metrics, store to DB, etc.
        except Exception:
            pass
            # if a hook crashes, silently ignore the error
            # we NEVER want a logging hook failure to crash the actual evaluation
            # "pass" = do nothing = swallow the exception completely
