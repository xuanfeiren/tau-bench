#!/usr/bin/env sh
set -eu

# Determine script directory
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# Ensure uv is installed
if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
else
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is available on PATH for this shell session
if ! command -v uv >/dev/null 2>&1; then
  if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
  hash -r 2>/dev/null || true
fi

# Locate uv binary
if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found after installation; ensure \$HOME/.local/bin is on PATH"
  exit 1
fi

# Clone algorithm repos if not present (needed as uv editable sources)
echo ""
echo "Checking algorithm repos..."

clone_or_verify() {
  local name="$1"
  local url="$2"
  local branch="${3:-}"
  local target="$SCRIPT_DIR/$name"

  if [ -d "$target" ]; then
    echo "  ✓ $name already present"
    # If a specific branch is required, verify and switch if needed
    if [ -n "$branch" ]; then
      current_branch=$(cd "$target" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
      if [ "$current_branch" != "$branch" ]; then
        echo "    ⚠ $name is on branch '$current_branch', switching to '$branch'..."
        (cd "$target" && git fetch origin && git checkout "$branch")
        echo "    ✓ $name now on branch '$branch'"
      else
        echo "    ✓ $name is on correct branch '$branch'"
      fi
    fi
  else
    echo "  ⏳ Cloning $name..."
    git clone "$url" "$target"
    if [ -n "$branch" ]; then
      echo "    Checking out branch: $branch"
      (cd "$target" && git checkout "$branch")
    fi
    echo "  ✓ $name cloned"
  fi
}

clone_or_verify "Trace"     "https://github.com/xuanfeiren/Trace.git"     "experimental"
clone_or_verify "dspy-repo" "https://github.com/xuanfeiren/dspy-repo.git"
clone_or_verify "gepa-repo" "https://github.com/xuanfeiren/gepa-repo.git"
clone_or_verify "openevolve" "https://github.com/xuanfeiren/openevolve.git"

# Install Python dependencies with uv
echo ""
echo "Installing Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Verify key imports
echo ""
echo "Verifying installation..."
python -c "import tau_bench; print('  tau_bench: OK')"
python -c "import opto; print('  opto (Trace): OK')"
python -c "import dspy; print('  dspy: OK')"
python -c "import gepa; print('  gepa: OK')"
python -c "from openevolve.api import run_evolution; print('  openevolve: OK')"
python -c "import litellm; print('  litellm: OK')"

echo ""
echo "=== Installation Complete ==="
ACTIVATE_CMD="source \"$SCRIPT_DIR/.venv/bin/activate\""
echo "To activate this environment later, run: $ACTIVATE_CMD"
echo "Or simply use: uv run python <script.py>"
