import time
import logging
import functools

logger = logging.getLogger(__name__)


def retry_with_backoff(fn=None, max_retries=3, base_delay=1.0, max_delay=30.0,
                       exceptions=(Exception,), retryable_exceptions=None):
    """Retry a function with exponential backoff.

    Can be used as a decorator or called directly.

    As a decorator:
        @retry_with_backoff(exceptions=(ConnectionError, TimeoutError))
        def my_func():
            ...

    Direct call:
        result = retry_with_backoff(my_func, retryable_exceptions=(ConnectionError,))

    Args:
        fn: Callable to retry (when used as direct call)
        max_retries: Maximum number of retries (0 = no retry)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exceptions: Tuple of exception types to retry on (decorator style)
        retryable_exceptions: Alias for exceptions (direct call style)

    Returns:
        Result of fn() or decorated function

    Raises:
        Last exception if all retries exhausted
    """
    # Support both parameter names for backward compat
    retry_on = retryable_exceptions or exceptions

    def _execute(func):
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return func()
            except retry_on as e:
                last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt + 1, max_retries + 1, str(e), delay
                    )
                    time.sleep(delay)
        raise last_exception

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _execute(lambda: func(*args, **kwargs))
        return wrapper

    # Direct call: retry_with_backoff(fn, ...)
    if fn is not None:
        return _execute(fn)

    # Decorator: @retry_with_backoff(exceptions=...)
    return _decorator
