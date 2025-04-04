# Chat guidelines

When running vscode tasks in quick succession please don't explain them or your process until you finished running them all, and even when all tasks end please be concise with your explanations.

# Dependencies

Use the `uv add ${some-package}` shell command to add dependencies to the project (in the pyproject.toml).  
Use the `uv remove ${some-package}` shell command to remove dependencies from the project (in the pyproject.toml).

# Code generation Guidelines

Use the `Pre-Commit All` task to sync the venv, apply formatting, run tests and to check yourself.  
After every feature implementation test yourself using pytest tests in the `tests` directory, run the tests with the `Pre-Commit All` command.  
When the `Pre-Commit All` task fails, next time try to run it with the shell tool using `pre-commit run --all-files`.

# Git

Use the `Git add all` task after creating new files to add them to the `pre-commit` context.  
After making significant changes, run the `Pre-commit All`, then `Git add all` if there are any changes, and then commit them.  
Commit messages must follow the pattern "<type>: <sentence>\n[<details>]", where the <type> is one of [feat, fix], the <sentence> is no more than 60 characters and the <details> are optional.  
Use the `Git push` task after every successful commit on an existing branch.

# Environment

Since this is a devcontainer, when making changes to the environment please make sure to add the new installations and setup to an automated script or a persistent tool, such as the devcontainer dockerfile, or using the `uv add/remove` shell commands etc.
