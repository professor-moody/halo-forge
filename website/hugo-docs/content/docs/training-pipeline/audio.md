---
title: "Audio Training"
description: "Audio and speech training paths"
weight: 18
---

The verified guided audio path currently covers Whisper-style speech-to-text
(ASR) adaptation. Audio classification and TTS may have loaders or verifier
components in the codebase, but they do not yet have verified guided
data-to-weight-update contracts and are shown as unavailable.

## Dashboard

Open **Train**, choose **Audio**, then choose **Speech recognition (ASR)**.
Preflight reports missing decoder dependencies or unsupported model families;
classification and TTS cannot be selected in Guided mode.

## CLI

```bash
halo-forge audio train --dataset librispeech --model openai/whisper-small --task asr --output ~/.halo-forge/runs/audio-asr
```

Audio runs are dependency-sensitive; install the audio extras before scaling beyond smoke tests.
