# Repository Instructions

This is a public, reusable multi-skill repository.

- Keep skills generic. Do not personalize instructions to a specific person, company, machine, or workspace.
- Refer to the human as "the user" or "user", not by name.
- Use relative paths inside skills. Avoid absolute local paths and OS-specific assumptions unless clearly documented as optional setup.
- Write skills so they work across different users, computers, and operating systems where practical.
- Keep each skill self-contained with `SKILL.md`, optional `scripts/`, and optional `references/`.
- For skills that need credentials or env vars, include a setup script that prompts locally and saves config under `~/.config/<skill-name>/`, like `telegram-cli/scripts/setup-api-key.sh`.
- Before every commit, check `git status --ignored --short` and confirm secrets, local config, caches, logs, virtualenvs, and generated junk are ignored and not staged.
