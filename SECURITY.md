# Security policy

## Supported versions

Security fixes go into the latest release. Please update to the
[latest release](https://github.com/AverWasTaken/localflow/releases/latest) before reporting.

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

1. Go to the [Security tab](https://github.com/AverWasTaken/localflow/security) of this repository.
2. Click **Report a vulnerability**.
3. Describe the problem, the LocalFlow version, and the steps to reproduce it.

Only the maintainer can see these reports. The maintainer will reply as soon as they can. Once a fix is released the
report can be published as a GitHub security advisory, with credit to you if you want it.

## Scope

Things that are in scope include:

- The installer and the install, uninstall and startup-task scripts (`install.ps1`, `uninstall.ps1`, `scripts/`,
  `installer/`), including the elevated startup task.
- Anything that could make LocalFlow send audio or text off your computer, or expose the local llama.cpp server
  (`127.0.0.1:8765`) beyond your machine.
- Code paths that read `config.json`, the history database or downloaded model files.

The startup task runs LocalFlow with your highest rights from files in your user folder. This is documented in the
README; reports that depend only on an attacker already being able to change files in your user account are known
and out of scope unless they show a new way in.

Please do not include real dictation text or personal data in a report unless it is needed to show the problem.
