---
applyTo: '**'
---

# General Information

This python (3.13) repo uses the astral.sh stack along other tools:
1. `pre-commit` - local automation
2. `uv` - venv and tools management
3. `ruff` - format and lint
4. `ty` - type checking
5. `pytest` - testing
6. `typos` - spell checking
7. `yamlfmt` - yaml format and lint
8. `biomejs` - json format and lint
9. `rumdl` - markdown format and lint
10. `taplo` - toml format and lint

# Code generation Guidelines

Use the `pre-commit run --all-files` shell command to sync the venv, apply formatting, run tests and to check yourself.  
When you are asked to "check" a file, it means you should run `pre-commit`.

# Git

Use the `Git add all` task after creating new files to add them to the `pre-commit` context.  
After making significant changes, run the `pre-commit run --all-files`, then `Git add all` if there are any changes, and then commit them.  
Commit messages must follow the pattern "<type>: <sentence>\n[<details>]", where the <type> is one of [feat, fix], the <sentence> is no more than 60 characters and the <details> are optional.  
Use the `Git push` task after every successful commit on an existing branch.
