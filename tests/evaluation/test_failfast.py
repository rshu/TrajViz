"""Empty-input fail-fast: a misconfigured data root must error loudly, never emit a
clean all-green scorecard.

Run:  python3 -m unittest tests.test_failfast -v
"""

import unittest

from agent_trajviz.awe._engine import config, digest


class TestRequireInstances(unittest.TestCase):
    def test_empty_raises_with_path(self):
        with self.assertRaises(ValueError) as cm:
            config.require_instances([])
        self.assertIn(str(config.ARGUS_ROOT), str(cm.exception))

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            config.require_instances(None)

    def test_nonempty_passthrough(self):
        self.assertEqual(config.require_instances(["i1", "i2"]), ["i1", "i2"])


class TestBuildersGuard(unittest.TestCase):
    def test_build_digest_empty_raises(self):
        # the guard is the first line, so this raises before any data access
        with self.assertRaises(ValueError):
            digest.build_digest([])


if __name__ == "__main__":
    unittest.main()
