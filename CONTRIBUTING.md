# Contributing Guide

Thanks for contributing to Adverse Vision MVP. This file explains the expected workflow for pull requests.

## 1) Local Setup

```powershell
cd "D:\Soft computing"
.\soft_comp\Scripts\activate
python -m pip install -r requirements.txt
```

## 2) Branching

- Create a short, descriptive branch name.
- Suggested naming:
  - `feat/<short-name>`
  - `fix/<short-name>`
  - `docs/<short-name>`
  - `test/<short-name>`

## 3) Coding Expectations

- Keep functions focused and readable.
- Add or update tests for behavior changes.
- Keep docs updated if CLI flags, APIs, or outputs change.
- Do not commit generated artifacts or local virtual environments.

## 4) Pre-PR Checklist

Run before opening a pull request:

```powershell
python -m pytest -q
```

If you changed dependency metadata:

```powershell
python -m pip install -r requirements.txt
```

## 5) Pull Request Requirements

- Explain the problem and the fix clearly.
- Include test evidence (output or description).
- Mention any known limitations or follow-up work.
- Keep PRs focused; avoid mixing unrelated changes.

## 6) Commit Message Suggestions

Examples:
- `feat: add ONNX benchmark output to CPU report`
- `fix: make dataset metadata batch-collatable`
- `docs: expand beginner quickstart troubleshooting`

## 7) First-Time Contributors

Good starter tasks:
- improve docs clarity
- add edge-case tests
- improve logging and error messages
