# GitHub Setup (Beginner)

This guide helps you publish this project to GitHub from `D:\Soft computing`.

## 1) Confirm Git Repo Is Initialized

```powershell
cd "D:\Soft computing"
git status
```

You should see files listed as untracked or staged.

## 2) Stage and Commit

```powershell
git add .
git commit -m "chore: initialize adverse vision MVP project"
```

## 3) Create an Empty GitHub Repository

On GitHub:
1. Click **New repository**
2. Name it (example: `adverse-vision-mvp`)
3. Do **not** initialize with README/license/gitignore (already present locally)
4. Create repository

## 4) Connect Local Repo To GitHub

Replace `<YOUR_GITHUB_URL>` with your repo URL:

```powershell
git remote add origin <YOUR_GITHUB_URL>
git push -u origin main
```

Example URL formats:
- `https://github.com/<username>/adverse-vision-mvp.git`
- `git@github.com:<username>/adverse-vision-mvp.git`

## 5) Typical Daily Workflow

```powershell
git checkout -b feat/my-change
# make changes
python -m pytest -q
git add .
git commit -m "feat: describe your change"
git push -u origin feat/my-change
```

Then open a Pull Request on GitHub.

## 6) Troubleshooting

If push fails with authentication errors:
- Use GitHub CLI (`gh auth login`) or a Personal Access Token for HTTPS.

If you accidentally tracked large files:
- Update `.gitignore`
- Untrack with `git rm --cached <path>`
- Commit again.
