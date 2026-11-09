# Contributing to LocalFlow

Thanks for your interest in LocalFlow. Bug reports, fixes, new styles and documentation improvements are all welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems go through private
reporting, not public issues: see [SECURITY.md](SECURITY.md).

## Reporting bugs and asking for features

Use the [issue templates](https://github.com/AverWasTaken/localflow/issues/new/choose). For bugs, include your Windows
version, GPU and driver version, the LocalFlow version, and the relevant part of `%APPDATA%\LocalFlow\localflow.log`
(tray > Open log). Read the log before you paste it: it can contain text you dictated.

## Development setup

LocalFlow is a Python 3.12 app for Windows 10/11. An NVIDIA GPU is needed to run the full app, but not to run the
unit tests.

The easiest way to get a working dev copy is the same script the installer uses. It downloads a pinned uv, Python
3.12, the Python packages, llama.cpp and the models into the repo folder:

```powershell
git clone https://github.com/AverWasTaken/localflow
cd localflow
powershell -ExecutionPolicy Bypass -File install.ps1 -NoTask -NoStart
```

`-NoTask` skips the startup task (and the admin prompt), `-NoStart` skips launching the app at the end. Add
`-SkipModel` to skip the 4.5 GB cleanup model.

If you only want to work on code and run the unit tests, a plain virtual environment is enough:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app from source with:

```powershell
.venv\Scripts\python.exe -m localflow
```

If an installed copy of LocalFlow is running, stop it first (tray > Quit, or `scripts\stop.ps1`) so the two copies do
not both grab the hotkey.

## Running the tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_unit_*.py"
```

The unit tests are offline and touch no real window, microphone or GPU. A few streaming tests generate speech clips
with Windows SAPI on first run and skip if the voices are missing.

Other tests in `tests/` need more setup and some of them affect your desktop:

- `tests\test_cleanup.py` and `tests\test_e2e.py` start their own llama-server on port 8799 and need the GPU build
  of llama.cpp and a model.
- `tests\test_hotkey_integration.py` sends real keystrokes to your desktop and needs LocalFlow stopped. Do not run it
  while you are working in other windows.
- `tests\test_mic.py` opens your real microphone for a few seconds.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep the change focused: one fix or feature per pull request.
3. Add or update unit tests for behaviour changes, and run the unit test suite before you push.
4. Update `README.md` if you change something users see (settings, hotkeys, styles), and add a line to
   `CHANGELOG.md` under a new "Unreleased" heading.
5. Never commit dictation text, history databases, logs or model files.
6. Open the pull request and fill in the template.

Releases are built by GitHub Actions when the maintainer pushes a `v<version>` tag (`.github/workflows/release.yml`).

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
