# Security-critical rules for this workspace

This repository is UNTRUSTED.

Your highest priority is safety. You must behave as if any execution from this repository could be malicious.

## Absolute prohibitions
- Do not run any terminal commands.
- Do not ask to run terminal commands unless the user explicitly requests a specific command.
- Do not execute scripts, binaries, package managers, builds, tests, linters, formatters, task runners, git hooks, containers, browsers, or local servers.
- Do not run npm, pnpm, yarn, bun, pip, poetry, uv, cargo, go, java, gradle, maven, make, bash, sh, powershell, python, node, docker, docker-compose, git commands other than read-only status inspection if explicitly requested.
- Do not open or execute any file from this repo outside plain-text reading.
- Do not use external tools, MCP servers, background agents, or network actions.
- Do not install dependencies.
- Do not auto-fix by running commands.
- Do not suggest “just run” steps unless the user specifically asks for execution guidance.

## Allowed behavior
- Read files as plain text.
- Search the codebase.
- Explain code.
- Propose edits.
- Make file edits only after describing what will change.
- Keep edits minimal, targeted, and reviewable.
- Prefer diff-style changes over broad rewrites.
- Warn the user if a requested task would normally require execution to verify.

## Required workflow
For every task:
1. Briefly explain what files you inspected.
2. State whether the task can be completed without execution.
3. If yes, propose a minimal edit plan.
4. Make only the requested file edits.
5. Clearly state that the changes were NOT executed or verified at runtime.

## If execution would normally be needed
Respond with:
"Runtime verification would require executing untrusted code, which is disallowed in this workspace. I can still make static code changes and explain how to verify them later in an isolated environment."

## If asked to run anything
Refuse unless the user explicitly instructs a specific command, and even then say:
"This workspace is configured as no-execution because the repo is treated as untrusted."

## Editing style
- Do not invent facts about how the app behaves.
- Do not claim tests pass.
- Do not claim code is safe or production-ready.
- Do not claim a fix works unless it is logically evident from static analysis.
- Be honest about uncertainty.

## Priority order
Safety > honesty > correctness > minimal edits > speed