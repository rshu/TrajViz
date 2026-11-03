"""Bounded argv rules for non-Python test *intent*, not proof of execution.

Keep option operands out of command/target matching. Unknown wrappers, aliases,
package scripts and build configuration still require benchmark-specific review.
"""


def _command(args, values=(), flags=(), toolchain=False):
    i = 0
    while i < len(args):
        a = args[i]
        if toolchain and i == 0 and a.startswith("+"):
            i += 1
        elif a.split("=", 1)[0] in values:
            i += 1 if "=" in a else 2
        elif a in flags:
            i += 1
        elif a.startswith("-"):
            return "", []  # unknown global option: do not guess its operands
        else:
            return a, args[i + 1 :]
    return "", []


def _positionals(args, values=()):
    """Targets/phases with the supported value-taking options removed."""
    result = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            result.extend(args[i + 1 :])
            break
        if a.split("=", 1)[0] in values:
            i += 1 if "=" in a else 2
        else:
            if not a.startswith("-"):
                result.append(a)
            i += 1
    return result


def _go_executes(args):
    values = {
        "-run",
        "-bench",
        "-timeout",
        "-count",
        "-p",
        "-tags",
        "-o",
        "-exec",
        "-ldflags",
        "-gcflags",
        "-asmflags",
        "-coverpkg",
        "-coverprofile",
        "-outputdir",
    }
    i = 0
    while i < len(args):
        a = args[i]
        name, sep, value = a.partition("=")
        if name in ("-list", "-test.list"):
            return False
        if name in ("-c", "-n") and (not sep or value.lower() == "true"):
            return False
        if name in values and not sep:
            i += 2
        else:
            i += 1
    return True


def _maven_skips(args):
    properties = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-D" and i + 1 < len(args):
            i += 1
            prop = args[i]
        elif a.startswith("-D"):
            prop = a[2:]
        else:
            i += 1
            continue
        name, sep, value = prop.partition("=")
        properties[name] = value.lower() if sep else "true"
        i += 1
    return any(properties.get(k) == "true" for k in ("skipTests", "maven.test.skip"))


def nonpython_test_command(head, args):
    if head == "go":
        command, tail = _command(args, values={"-C"})
        return command == "test" and _go_executes(tail)
    if head == "cargo":
        command, tail = _command(
            args,
            values={"--config", "--color", "-C"},
            flags={"-v", "-vv", "--verbose", "-q", "--quiet", "--locked", "--offline", "--frozen"},
            toolchain=True,
        )
        cargo_args = tail[: tail.index("--")] if "--" in tail else tail
        harness_args = tail[tail.index("--") + 1 :] if "--" in tail else []
        return command == "test" and "--no-run" not in cargo_args and "--list" not in harness_args
    if head in ("npm", "pnpm", "yarn", "bun"):
        values = {"-w", "--workspace", "--prefix", "--cwd", "--filter", "-C", "--dir", "--cache"}
        flags = {
            "--silent",
            "-s",
            "--workspaces",
            "--if-present",
            "--recursive",
            "-r",
            "--offline",
            "--no-color",
            "--ignore-scripts",
        }
        command, tail = _command(args, values, flags)
        if command in ("workspace",) and head == "yarn" and tail:
            command, tail = _command(tail[1:], values, flags)
        if command in ("run", "run-script"):
            command, tail = _command(tail, values, flags)
            return command == "test" or command.startswith("test:")
        return command in ("test", "t", "tst") or (head != "npm" and command.startswith("test:"))
    if head in ("mvn", "mvnw"):
        if _maven_skips(args):
            return False
        phases = _positionals(
            args,
            {
                "-f",
                "--file",
                "-pl",
                "--projects",
                "-s",
                "--settings",
                "-gs",
                "--global-settings",
                "-T",
                "--threads",
                "-rf",
                "--resume-from",
                "-D",
            },
        )
        return bool(set(phases) & {"test", "verify", "integration-test"})
    if head in ("gradle", "gradlew"):
        if "-m" in args:
            return False
        values = {
            "-p",
            "--project-dir",
            "-b",
            "--build-file",
            "-c",
            "--settings-file",
            "-I",
            "--init-script",
            "--tests",
            "-x",
            "--exclude-task",
        }
        targets = _positionals(args, values)
        excluded = set()
        for i, a in enumerate(args):
            if a in ("-x", "--exclude-task") and i + 1 < len(args):
                excluded.add(args[i + 1].rsplit(":", 1)[-1])
            elif a.startswith("--exclude-task="):
                excluded.add(a.split("=", 1)[1].rsplit(":", 1)[-1])
        for target in targets:
            task = target.rsplit(":", 1)[-1]
            if task in excluded or (task == "check" and "test" in excluded):
                continue
            lower = task.lower()
            if lower in {"test", "tests", "check", "integrationtest", "functionaltest", "unittest", "e2etest"}:
                return True
            if (lower.startswith("test") and lower.endswith("unittest")) or (
                lower.startswith("connected") and lower.endswith("androidtest")
            ):
                return True
        return False
    if head in ("make", "gmake", "ninja"):
        noexec = {"-n", "--just-print", "--recon", "-q", "--question", "-t", "--touch"}
        if any(a in noexec for a in args):
            return False
        targets = _positionals(args, {"-C", "--directory", "-f", "--file", "--makefile"})
        return any(a in {"test", "tests", "check"} for a in targets)
    if head == "ctest":
        return not any(a == "-N" or a.split("=", 1)[0] == "--show-only" for a in args)
    if head == "meson":
        command, tail = _command(args)
        return command == "test" and "--list" not in tail
    if head in ("dotnet", "bazel", "buck"):
        command, _ = _command(args, values={"--output_base", "--output_user_root"}, flags={"--batch"})
        return command == "test"
    if head == "ant":
        return "test" in _positionals(args, {"-f", "-file", "-buildfile", "-lib", "-logfile"})
    if head in ("jest", "vitest", "mocha", "rspec", "phpunit", "jtreg"):
        return not (head == "vitest" and args[:1] == ["list"])
    if head in ("npx", "pnpx"):
        command, tail = _command(
            args, values={"-p", "--package", "--cache", "--prefix"}, flags={"-y", "--yes", "--no-install"}
        )
        return command in ("jest", "vitest", "mocha") and nonpython_test_command(command, tail)
    return False
