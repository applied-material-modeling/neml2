# AI Workflow Usage

This directory contains the cross-LLM wrapper structure for this repository.

Behavior is defined in [`AGENTS.md`](../AGENTS.md). Files under `ai/` are only named entry points
and indexes so users and LLMs can refer to the same workflows, skills, and checks consistently.

## Mental Model

The same repository should work across Claude Code, Codex, and Gemini:

- Claude: you may use `/{workflow}` as a shortcut
- Codex: ask it to execute a named workflow
- Gemini: ask it to read `AGENTS.md` and run a named workflow

The actual logic still comes from `AGENTS.md`.

## Workflow Shortcuts

In Claude Code, `/{workflow}` may be used as a convenience shortcut for any workflow listed below:

- `/build`
- `/fix-build`
- `/test`
- `/docs`
- `/docs-verify`
- `/implement`
- `/setup`
- `/naming`
- `/smoke-test`

These shortcuts are wrappers only. They must execute the matching workflow from `AGENTS.md` or
`ai/workflows/`, not invent a different process.

## Workflow Options

Available workflows:

- `BUILD`: configure and build the C++ backend
- `FIX-BUILD`: diagnose and repair a failing build
- `TEST`: run C++ and/or Python tests
- `DOCS`: add or improve documentation
- `DOCS-VERIFY`: build documentation and report errors or warnings
- `IMPLEMENT`: implement a model end-to-end
- `SETUP`: set up the Python development environment
- `NAMING`: audit naming consistency
- `SMOKE-TEST`: verify the workflow itself

Workflow files:

- [`ai/workflows/build.md`](./workflows/build.md)
- [`ai/workflows/fix-build.md`](./workflows/fix-build.md)
- [`ai/workflows/test.md`](./workflows/test.md)
- [`ai/workflows/docs.md`](./workflows/docs.md)
- [`ai/workflows/docs-verify.md`](./workflows/docs-verify.md)
- [`ai/workflows/implement.md`](./workflows/implement.md)
- [`ai/workflows/setup.md`](./workflows/setup.md)
- [`ai/workflows/naming.md`](./workflows/naming.md)
- [`ai/workflows/smoke-test.md`](./workflows/smoke-test.md)

## How To Use

### Claude Code

You may use workflow shortcuts:

```text
/build
/test
/implement SalehaniIrani3DCTraction
/docs src/neml2/models/solid_mechanics/...
```

### Codex

Use natural language and name the workflow explicitly:

```text
Read AGENTS.md first.
Then execute the IMPLEMENT workflow for SalehaniIrani3DCTraction.
```

```text
Follow AGENTS.md.
Run the TEST workflow for [tensors].
```

### Gemini

Use the same pattern explicitly:

```text
Read AGENTS.md first.
Run the BUILD workflow.
```

```text
Read AGENTS.md and ai/workflows/test.md.
Run the TEST workflow for python.
Simulate hooks explicitly using ai/hooks/.
```

## Skills And Hooks

- `ai/skills/`: reusable specialized behavior such as `CODE-WRITER`, `DOC-WRITER`,
  `TEST-WRITER`, and `BUILD-ENGINEER`
- `ai/hooks/`: explicit checks that replace old automatic hooks

These are not separate sources of truth. They point back to `AGENTS.md`.

