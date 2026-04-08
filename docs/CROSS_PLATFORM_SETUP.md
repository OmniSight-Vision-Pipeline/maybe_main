# Cross-Platform Setup

This project should be cloned the same way on every machine, but the virtual environment must be created locally on each operating system.

## Why The Venv Is Not In Git

The `soft_comp` environment is intentionally ignored by Git because:
- Windows virtual environments do not work on macOS/Linux.
- macOS/Linux virtual environments do not work on Windows.
- They contain platform-specific binaries, paths, and activation scripts.

The shared part is:
- source code
- `requirements.txt`
- docs

The local part is:
- virtual environment
- downloaded model weights
- generated artifacts

## Windows Setup

From the project root:

```powershell
.\scripts\setup_windows.ps1
.\soft_comp\Scripts\activate
```

If PowerShell blocks script execution, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

## macOS / Linux Setup

From the project root:

```bash
bash ./scripts/setup_unix.sh
source soft_comp/bin/activate
```

If `python3` is missing, install Python 3 first and retry.

## Verify Setup

After activation:

```bash
python -m pytest -q
```

## Important Note For Teams

Do not try to share one teammate's `soft_comp` folder across machines.  
Each teammate should clone the repo and run the setup script for their own OS.
