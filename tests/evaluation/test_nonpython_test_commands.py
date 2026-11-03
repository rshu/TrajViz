"""Executable/subcommand position and nonexecution modes matter across runners."""

import unittest

from agent_trajviz.awe._engine.trajectory import (
    EDIT,
    NormalizedStep,
    NormalizedTrajectory,
    classify_command,
    is_test_command,
    is_suite_wide_test_command,
)
from agent_trajviz.awe._engine.trajsignals import compute
from .test_batch8_signals import _gold


NEGATIVES = [
    "go run main.go test",
    "go build test",
    "go test -list . ./...",
    "go test -list=TestFix ./...",
    "go test -test.list=.",
    "go test -c ./...",
    "go test -c=true",
    "go test -n ./...",
    "go -C test run main.go test",
    "cargo run -- test",
    "cargo build -p test",
    "cargo test --no-run",
    "cargo test -- --list",
    "cargo +stable test --no-run",
    "npm install test",
    "npm --prefix test install foo",
    "npm test:unit",
    "npm run build -- test",
    "pnpm add test",
    "yarn install test",
    "bun add test",
    "mvn test -DskipTests",
    "mvn verify -DskipTests=true",
    "mvn test -Dmaven.test.skip=true",
    "mvn test -D skipTests",
    "mvn -f test clean",
    "mvn test -DskipTests=false -DskipTests=true",
    "ctest -N",
    "ctest --show-only",
    "ctest --show-only=json-v1",
    "gradle contest",
    "gradle latest",
    "gradle testClasses",
    "gradle testResults",
    "gradle -p test assemble",
    "gradle test -x test",
    "gradle check --exclude-task=test",
    "gradle test -m",
    "gradle assemble --tests test",
    "make -C test all",
    "make -f test all",
    "make latest",
    "make -n test",
    "make -q test",
    "ninja -n test",
    "meson compile test",
    "meson test --list",
    "dotnet build test",
    "bazel run test",
    "buck build test",
    "ant -f test compile",
    "npx --package jest echo test",
    "npx -p jest -- echo jest",
    "vitest list",
    "npx vitest list",
    'grep "go test" README.md',
    'echo "cargo test"',
]

POSITIVES = [
    "go test ./...",
    "go -C src test ./...",
    "go -C=src test ./...",
    "go test -c=false",
    "go test -n=false",
    'go test -run "-c" ./...',
    "cargo test --workspace",
    "cargo +stable test",
    "cargo --locked test",
    "cargo test -- --nocapture",
    "npm test",
    "npm --workspace app test",
    "npm --workspace=test run test",
    "npm run test:unit",
    "npm t",
    "pnpm --filter app test",
    "yarn workspace core test",
    "yarn test:unit",
    "bun test",
    "bun run test:unit",
    "mvn test",
    "./mvnw verify",
    "mvn test -DskipTests=false",
    "mvn test -Dmaven.test.skip=false",
    "mvn test -DskipTests=true -DskipTests=false",
    "mvn test -Dskiptests=true",
    "ctest --test-dir build",
    "gradle test",
    "./gradlew :core:test",
    "gradle integrationTest -x test",
    "gradle testDebugUnitTest",
    "gradle connectedDebugAndroidTest",
    "gradle check",
    "make check",
    "ninja test",
    "meson test",
    "dotnet test",
    "bazel --output_base build test //...",
    "buck test",
    "ant test",
    "npx --yes jest",
    "npx --package jest jest",
    "pnpx vitest run",
    "jest",
    "vitest run",
    "mocha",
    "rspec",
    "phpunit",
    "jtreg tests",
    "env CI=true timeout 30 go test ./...",
]


class TestNonPythonCommands(unittest.TestCase):
    def test_false_positive_counterexamples(self):
        for command in NEGATIVES:
            with self.subTest(command=command):
                self.assertFalse(is_test_command(command))
                self.assertFalse(is_suite_wide_test_command(command))

    def test_real_test_intent_is_preserved(self):
        for command in POSITIVES:
            with self.subTest(command=command):
                self.assertTrue(is_test_command(command))

    def test_sibling_segments_preserve_actual_test_intent(self):
        self.assertTrue(is_test_command("cargo test --no-run && cargo test"))
        self.assertTrue(is_test_command("npm install test; npm run test"))
        self.assertTrue(is_suite_wide_test_command("make -n test && make test"))

    def test_nonexecuting_commands_do_not_close_verification_or_repair_gaps(self):
        for command in NEGATIVES:
            steps = [
                NormalizedStep(0, 0, "agent", "tool_call", canonical_action=EDIT, files=["pkg/a.py"], call_id="e"),
                NormalizedStep(1, 0, "tool", "tool_result", call_id="e"),
                NormalizedStep(
                    2,
                    1,
                    "agent",
                    "tool_call",
                    canonical_action=classify_command(command),
                    test_run=is_test_command(command),
                    args={"command": command},
                    call_id="t",
                ),
                NormalizedStep(3, 1, "tool", "tool_result", call_id="t", args={"exit_code": 0}, output_excerpt="done"),
            ]
            with self.subTest(command=command):
                sig = compute(NormalizedTrajectory("a", "i", steps), _gold())
                self.assertFalse(sig.ran_any_test)
                self.assertFalse(sig.encountered_failure)
                self.assertTrue(sig.no_test_after_final_edit)
