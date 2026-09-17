# Project version recording

This project uses ResearchPilot `research[F]-iteration` for all version changes.

Before modifying code, configuration, manifests, protocol, analysis, or run scripts, read the local `docs/user_requirements.md`, `docs/dev_log.md`, and `docs/recharge_return_plan.md` when present. They are intentionally not published. In a fresh clone, use `experiments/recharge_return/protocol.md` and this file until local research notes are supplied. Preserve the pilot/formal evidence boundary.

Assign the next unique development version when a change is made. Append to `docs/dev_log.md`; never delete or rewrite earlier entries. Record the reason, exact files, protocol impact, validation commands and outcomes, results, failures, and limitations. Append a new result note after verification. For a multi-file change, record every changed file in the entry. Do not present smoke tests or reference-controller probes as learned-policy evidence.

For every completed version, add an immutable `docs/versions/<version>.json` containing SHA-256 hashes of the published source, config, manifests, scripts, tests, and protocol at that version. Historical versions without source snapshots must remain explicitly marked unrecoverable. Keep generated outputs, the local `.venv`, and the three unpublished research notes out of the source manifest.
