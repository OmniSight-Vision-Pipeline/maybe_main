#!/usr/bin/env bash
set -euo pipefail

VENV_NAME="${1:-soft_comp}"

echo "Creating virtual environment '${VENV_NAME}' on macOS/Linux..."
python3 -m venv "${VENV_NAME}"

PYTHON_BIN="${VENV_NAME}/bin/python"
if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Virtual environment python executable not found at ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Upgrading pip..."
"${PYTHON_BIN}" -m pip install --upgrade pip

echo "Installing project dependencies..."
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "Activate with:"
echo "source ${VENV_NAME}/bin/activate"
