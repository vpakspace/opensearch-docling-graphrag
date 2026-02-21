"""Tests for retry decorator."""

import httpx
import pytest

from opensearch_graphrag.retry import with_retry


def test_no_retry_on_success():
    calls = []

    @with_retry(max_retries=2, backoff_base=0.01)
    def ok():
        calls.append(1)
        return "done"

    assert ok() == "done"
    assert len(calls) == 1


def test_retry_on_connect_error():
    calls = []

    @with_retry(max_retries=2, backoff_base=0.01)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "recovered"

    assert flaky() == "recovered"
    assert len(calls) == 3


def test_exhaust_retries():
    @with_retry(max_retries=1, backoff_base=0.01)
    def always_fail():
        raise httpx.ReadTimeout("timeout")

    with pytest.raises(httpx.ReadTimeout):
        always_fail()


def test_non_retryable_propagates():
    @with_retry(max_retries=2, backoff_base=0.01)
    def value_err():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        value_err()
