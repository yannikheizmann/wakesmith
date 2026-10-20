# Security policy

## Supported versions

Until the first stable release, security fixes are applied to the latest revision of the
`main` branch.

## Reporting a vulnerability

Please do not report suspected vulnerabilities in a public issue. Use GitHub's private
vulnerability reporting for this repository, or contact Yannik Heizmann at
`research@yannikheizmann.com`. Include the affected commit, the component and execution
path, steps to reproduce, and the expected impact.

Do not include access tokens or private data in a report.

## Scope

wakesmith's source code, packaging, download and caching logic, and command-line workflow
are in scope. Vulnerabilities in third-party models, datasets or dependencies should be
reported to their maintainers unless wakesmith uses them in a way that creates a
project-specific risk. Downloads are pinned to fixed revisions; model checkpoints from
third parties are loaded with their libraries' own loaders.
