# Changelog

All notable changes to LocalFlow are listed here. Versions follow [Semantic Versioning](https://semver.org).

## [1.1.1] - 2026-11-08

### Fixed

- Setup no longer fails with error 448 ("untrusted mount point") for non-admin installs. Inno Setup 6.7 turns on
  RedirectionGuard by default, which broke creating the Python environment. Setup now turns it off (it runs as the
  user), requires Inno Setup 6.7 or newer, installs Python in its own step, and repairs a leftover empty folder or
  broken link from an earlier failed run.

## [1.1.0] - 2026-11-08

### Added

- Windows setup program (`LocalFlow-Setup-x.y.z.exe`, built with Inno Setup) that downloads everything LocalFlow needs
  during install. GitHub Actions builds it for `v*` tags.
- Cleanup model switcher in the tray and Settings, with a live server restart. Any `.gguf` file in the `models`
  folder shows up. Support for LFM3-1.7B (needs llama.cpp b11960 or newer).
- Command mode (Ctrl+Win): rewrite the selected text, or write at the cursor, by voice.
- Paste last transcript (Alt+Shift+Z), usage stats and history retention.
- Meeting recorder (your mic and the computer's audio as separate speakers) and audio/video file transcription, with
  a transcript store and local AI notes (summary, bullets, key points, action items, custom questions).
- New dark theme, Home page (stats and history) and Transcriptions page.
- `email` style: spoken greeting, body and closing laid out on separate lines for mail apps (Gmail, Outlook, Proton,
  Yahoo, Thunderbird).
- Game mode: frees the GPU while a configured game runs and reloads the models afterwards, with an optional CPU
  dictation fallback.
- Named or counted lists are formatted as bullets; list examples for the casual and professional styles.

### Changed

- The default cleanup model is Qwen3.5-4B at Q8_0 (was Q4_K_M).
- Casual style handles spoken code (in backticks), self-corrections, quotes and paragraphs better.
- Live dictation always takes priority over background GPU work.
- LLM context raised to 8192 tokens so long email dictations fit.
- `install.ps1` uses a pinned uv, a pinned llama.cpp build (b12211) checked against SHA-256 hashes, and can be re-run
  to repair an install. The install, uninstall and startup-task scripts only touch their own install folder.

## [1.0.0] - 2026-10-03

- Initial public release: push-to-talk dictation with faster-whisper and local LLM cleanup, per-app writing styles,
  streaming transcription, dictionary and snippets.

[1.1.1]: https://github.com/AverWasTaken/localflow/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/AverWasTaken/localflow/releases/tag/v1.1.0
[1.0.0]: https://github.com/AverWasTaken/localflow/commit/07c075e
