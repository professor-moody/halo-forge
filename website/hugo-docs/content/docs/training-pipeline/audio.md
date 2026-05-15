---
title: "Audio Training"
description: "Audio and speech training paths"
weight: 18
---

Audio training covers speech-to-text, classification, and related audio-language tasks.

## Dashboard

Open **Train**, choose **Audio**, then choose the audio method. Pick a task such as ASR before launching. Preflight reports missing dependencies or unsupported model families.

## CLI

```bash
halo-forge audio train --dataset librispeech --model openai/whisper-small --task asr --output ~/.halo-forge/runs/audio-asr
```

Audio runs are dependency-sensitive; install the audio extras before scaling beyond smoke tests.
