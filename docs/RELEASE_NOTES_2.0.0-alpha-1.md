# Halo Forge 2.0.0-alpha-1 Release Notes

Halo Forge 2.0.0-alpha-1 is the first app-first alpha. The CLI remains fully supported, but this release validates the dashboard and unsigned desktop app as the primary workstation surface.

> Distribution note: the macOS DMG in this prerelease is unsigned and may be rejected by Gatekeeper as damaged when downloaded from a browser. Use `2.0.0-alpha-2` or later for public macOS install testing.

## Highlights

- **Desktop app alpha**: unsigned Tauri macOS/Linux builds start a local Halo Forge dashboard on `127.0.0.1:8765`, show startup diagnostics, and use the bundled runtime path with source-checkout fallback.
- **Dashboard-first training**: Start stays beginner-safe, while Train exposes the broader method surface: SFT, RAFT, DPO, ORPO, RM, GRPO, VLM, audio, reasoning, and agentic workflows.
- **Managed serving**: Models and Results can start one local model server at a time, Playground defaults to the managed server, and Stop cleans up the dashboard-owned process.
- **Hugging Face access**: `/connect` now manages workstation-scoped HF tokens for gated/private model downloads without storing secrets in browser local storage.
- **Operator trust polish**: Start uses writable `~/.halo-forge/runs` defaults, gated-model errors are friendly, Liquid model links point at real Hugging Face pages, and Apple Silicon telemetry labels unavailable sensors clearly.
- **Live training monitor**: Run detail now shows stage progression, latest event, live loss history, elapsed time, artifact state, and classified failure summaries instead of inert charts or raw tracebacks.
- **Packaged-runtime training fix**: The desktop runtime handles PyInstaller multiprocessing helper processes correctly, and the alpha gate includes a packaged tiny SFT smoke that proves optimizer steps and final artifacts work outside the source `.venv`.

## Acceptance Focus

- Desktop launch and quit lifecycle.
- Start → run monitor → Results flow.
- Live monitor stages, log tail, real optimizer-step counting, and local-time timestamps.
- Models → Serve → Playground → Stop flow.
- Hugging Face token save/check/clear flow.
- Docs and release metadata reporting `2.0.0-alpha-1`.

## Out Of Scope

- Signed/notarized distribution.
- Auto-update.
- Windows packaging.
- Distributed workers, cloud queues, or multi-host scheduling.
