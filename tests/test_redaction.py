"""Credential-shaped text is redacted before any LLM request leaves the machine.

Trajectories routinely contain credentials an agent printed or typed: a key
in a curl command, a token in an export line, a private key it read. The
Issues judge and the step labeler send step text to the endpoint the user
configured, and nothing removed such strings first. Both egress functions now
pass every outgoing string through ``_redact.redact`` -- a safety net that
recognizes common key formats and assignments whose name says "secret",
while leaving ordinary code and prose alone.

Every secret below is synthetic.
"""

from __future__ import annotations

import unittest
from unittest import mock

from agent_trajviz._redact import PLACEHOLDER, redact

FAKE_OPENAI = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
FAKE_ANTHROPIC = "sk-ant-api03-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3Ss2Rr1"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_GITHUB = "ghp_" + "a" * 20 + "0123456789abcdef"
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" + "." + "dBjftJeZ4CVPmB92K27uhbUJU1p1rwW1gFWFOEjXk"
FAKE_PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA7\nabc123\n-----END RSA PRIVATE KEY-----"


class RedactTests(unittest.TestCase):
    def assertRedacted(self, text, secret):
        out, count = redact(text)
        self.assertNotIn(secret, out)
        self.assertGreaterEqual(count, 1)
        return out

    def assertUntouched(self, text):
        self.assertEqual(redact(text), (text, 0))

    def test_provider_key_formats(self):
        for secret in (FAKE_OPENAI, FAKE_ANTHROPIC, FAKE_AWS, FAKE_GITHUB, FAKE_JWT):
            with self.subTest(secret=secret[:8]):
                out = self.assertRedacted(f"curl -H 'k: {secret}' https://api.example.com", secret)
                self.assertIn(PLACEHOLDER, out)

    def test_private_key_block(self):
        out = self.assertRedacted(f"cat id_rsa\n{FAKE_PEM}\ndone", "MIIEpAIBAAKCAQEA7")
        self.assertIn("done", out)

    def test_secret_named_assignments(self):
        cases = [
            ("export OPENAI_API_KEY=abc123def456ghi789", "abc123def456ghi789"),
            ('DB_PASSWORD: "s3cretPassw0rd!x"'.replace("!", ""), "s3cretPassw0rdx"),
            ('{"client_secret": "9f8e7d6c5b4a39281706"}', "9f8e7d6c5b4a39281706"),
            ("--api-key 1234567890abcdefXYZ", "1234567890abcdefXYZ"),
            ("Authorization: Bearer abcDEF123456ghiJKL789", "abcDEF123456ghiJKL789"),
            ("git clone https://alice:hunter2pass99@github.com/x/y.git", "hunter2pass99"),
        ]
        for text, secret in cases:
            with self.subTest(text=text):
                out = self.assertRedacted(text, secret)
                # The name stays, so the step still reads as what it did.
                self.assertTrue(any(word in out for word in ("API_KEY", "PASSWORD", "secret", "api-key",
                                                             "Bearer", "alice")))

    def test_ordinary_code_is_left_alone(self):
        for text in (
            "max_tokens: 4096",
            "token_usage = response.usage.total_tokens",
            "password = os.environ['DB_PASSWORD']",
            "def get_secret(name): return vault.read(name)",
            "git checkout -b fix/token-refresh",
            "sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "See https://github.com/org/repo/blob/main/src/auth.py#L42",
            "The tokens_per_second metric divides by wall-clock time.",
        ):
            with self.subTest(text=text):
                self.assertUntouched(text)

    def test_a_url_password_is_redacted_whatever_it_looks_like(self):
        out = self.assertRedacted("pip install --index-url https://bob:letmein@pypi.example.com/simple x",
                                  "letmein")
        self.assertIn("https://bob:[REDACTED]@pypi.example.com", out)

    def test_pathological_input_stays_fast(self):
        import time

        # Sizes far beyond a real step; a quadratic pattern takes minutes here.
        for text in ("a." * 100_000, "token.a." * 25_000, "x" * 500_000, "secret" * 50_000,
                     "http" + ":" * 200_000, "--token " * 25_000,
                     "-----BEGIN RSA PRIVATE KEY-----" + "A" * 500_000):
            with self.subTest(text=text[:12]):
                start = time.perf_counter()
                redact(text)
                self.assertLess(time.perf_counter() - start, 2.0)

    def test_counts_every_redaction(self):
        _, count = redact(f"{FAKE_OPENAI} and {FAKE_AWS}")
        self.assertEqual(count, 2)


class ConfigReprTests(unittest.TestCase):
    def test_the_api_key_never_appears_in_the_config_repr(self):
        from agent_trajviz.insight.llm_config import AnalysisLLMConfig

        config = AnalysisLLMConfig(base_url="https://llm.example", api_key="sk-not-a-real-key-0123456789",
                                   model="m", provider="openai", temperature=None, max_tokens=1, timeout=1,
                                   source="analyze")
        self.assertNotIn("sk-not-a-real-key-0123456789", repr(config))
        self.assertNotIn("sk-not-a-real-key-0123456789", str(config))
        self.assertIn("model='m'", repr(config))


class EgressTests(unittest.TestCase):
    """Both functions that send HTTP requests redact what they send."""

    def _capture_post(self):
        sent = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}],
                        "content": [{"type": "text", "text": "ok"}]}

        def post(url, json=None, headers=None, timeout=None, allow_redirects=True):
            assert allow_redirects is False
            sent["body"] = json
            sent["headers"] = headers
            return _Resp()

        return sent, post

    def test_assistant_requests_are_redacted(self):
        import requests

        from agent_trajviz.insight import assistant
        from agent_trajviz.insight.llm_config import AnalysisLLMConfig

        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                cfg = AnalysisLLMConfig(base_url="https://llm.example.com", api_key="REAL-KEY-NOT-REDACTED",
                                        model="m", provider=provider, temperature=None, max_tokens=64,
                                        timeout=5, source="analyze")
                sent, post = self._capture_post()
                with mock.patch.object(requests, "post", post):
                    assistant.complete_chat(cfg, f"system {FAKE_AWS}",
                                            [{"role": "user", "content": f"why did {FAKE_OPENAI} fail?"}])
                body = str(sent["body"])
                self.assertNotIn(FAKE_AWS, body)
                self.assertNotIn(FAKE_OPENAI, body)
                # The user's own key still authenticates the request.
                self.assertIn("REAL-KEY-NOT-REDACTED", str(sent["headers"]))

    def test_labeler_requests_are_redacted(self):
        import requests

        from agent_trajviz.labeling import llm

        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                sent, post = self._capture_post()
                with mock.patch.object(requests, "post", post):
                    llm.call_llm("https://llm.example.com", "REAL-KEY", "m", f"sys {FAKE_GITHUB}",
                                 f"step: export TOKEN={FAKE_JWT}", provider=provider)
                body = str(sent["body"])
                self.assertNotIn(FAKE_GITHUB, body)
                self.assertNotIn(FAKE_JWT, body)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
