import unittest
from http.client import IncompleteRead
from unittest.mock import patch

from agent_trajviz.awe._engine import llm


class TestInterruptedResponse(unittest.TestCase):
    def test_interrupted_read_retries_before_returning_valid_completion(self):
        response = {"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]}
        with (
            patch.object(llm.config, "OPENROUTER_API_KEY", "test"),
            patch.object(llm, "_post", side_effect=[IncompleteRead(b"partial"), response]) as post,
            patch.object(llm, "_backoff") as backoff,
        ):
            result = llm.chat([{"role": "user", "content": "x"}], retries=2)
        self.assertEqual(result["content"], "complete")
        self.assertEqual(post.call_count, 2)
        backoff.assert_called_once_with(0)

    def test_exhausted_reads_raise_batch_driver_error(self):
        with (
            patch.object(llm.config, "OPENROUTER_API_KEY", "test"),
            patch.object(llm, "_post", side_effect=IncompleteRead(b"partial")),
            patch.object(llm, "_backoff"),
            self.assertRaisesRegex(llm.LLMError, "IncompleteRead"),
        ):
            llm.chat([{"role": "user", "content": "x"}], retries=2)
