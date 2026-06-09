# Repository Guidelines

## Project Structure & Module Organization
- `app.py` is the Gradio entrypoint and should stay runnable as `python app.py`.
- `pyproject.toml` and `uv.lock` define Python dependencies and locked versions.
- `requirements.txt` is generated for Hugging Face Spaces compatibility.
- `spec/` contains product and phase notes; use it to define behavior before implementing major features, but keep specs separate from runtime code.
- `.github/workflows/deploy-space.yml` syncs `main` to the hosted Space.

## Build, Test, and Development Commands
- `uv run python app.py` starts the local Gradio app.
- `./export_space_requirements.sh` refreshes the Space dependency file after dependency changes.
- `git pull` keeps `main` current before starting work.

## Coding Style & Naming Conventions
- Use Python 3.10+ syntax and standard library types like `str` in annotations.
- Keep functions small and explicit; prefer clear names such as `greet_rehearsal`.
- Use 4-space indentation, ASCII-only text unless the file already uses Unicode, and module-level constants only when they improve readability.
- No formatter or linter is configured yet, so follow the surrounding style closely.

## Testing Guidelines
- There is no automated test suite in the repository yet.
- For now, validate changes by running `uv run python app.py` and checking the interface manually.
- If you add tests, place them in a `tests/` directory and use descriptive names like `test_app.py`.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit messages and merge commits from PRs.
- Keep commits focused on one change, for example `Add speech rehearsal prompt`.
- PRs should summarize the user-visible change, mention any deployment impact, and link related issues or specs when relevant.
- Include screenshots or short notes for UI changes, and update `requirements.txt` when dependencies change.

## Agent-Specific Instructions
- Follow spec-driven development for meaningful features: write or update the relevant spec in `spec/` before implementing code changes.
- Treat specs as lightweight source-of-truth for intended behavior, acceptance criteria, and phase scope; update them when implementation decisions materially change.
- Do not edit generated files by hand if they can be regenerated from `uv`.
- Keep `main` deployable: anything merged there will sync to the Hugging Face Space.
