---
title: "Windows Build Server"
description: "Configure a Windows machine for MSVC verification"
weight: 4
---

This guide covers setting up a Windows machine for halo-forge's MSVC verification.

## Overview

The MSVC verifier compiles and optionally runs code on a remote Windows server via SSH. This enables:
- **Full MSVC compatibility** - Test with the real Windows compiler
- **Execution verification** - Run binaries and check output
- **Binary caching** - Save compiled executables for analysis

## Prerequisites

- Windows 10/11 (64-bit)
- Administrative access
- Network connectivity to Linux training host

## Step 1: Install OpenSSH Server

```powershell
# Install OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and enable the service
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Configure firewall
New-NetFirewallRule -Name 'OpenSSH-Server' -DisplayName 'OpenSSH Server' `
    -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

## Step 2: Configure SSH Key Authentication

On your **Linux** host:

```bash
# Generate SSH key if needed
ssh-keygen -t ed25519 -f ~/.ssh/win

# Copy public key to Windows
ssh-copy-id -i ~/.ssh/win.pub user@windows-ip
```

On **Windows**, ensure authorized_keys is set up:

```powershell
# Create .ssh directory
mkdir C:\Users\$env:USERNAME\.ssh

# Add public key (paste from Linux: cat ~/.ssh/win.pub)
notepad C:\Users\$env:USERNAME\.ssh\authorized_keys
```

## Step 3: Install Visual Studio Build Tools

Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/):

1. Run the installer
2. Select **Desktop development with C++**
3. Ensure these components are selected:
   - MSVC v143 - VS 2022 C++ x64/x86 build tools
   - Windows SDK (latest)
   - C++ CMake tools (optional)

### Configure MSVC in PowerShell Profile

Add to `$PROFILE` (usually `Documents\PowerShell\Microsoft.PowerShell_profile.ps1`):

```powershell
# Load MSVC environment on shell start
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

if (Test-Path $vcvarsPath) {
    cmd /c "`"$vcvarsPath`" && set" | ForEach-Object {
        if ($_ -match "^(.+?)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
    Write-Host "MSVC environment loaded" -ForegroundColor Green
}
```

Verify:

```powershell
cl.exe
# Should show: Microsoft (R) C/C++ Optimizing Compiler...
```

## Step 4: Create Working Directories

```powershell
# Create directories for file transfer
mkdir C:\Binaries\input   # Source files go here
mkdir C:\Binaries\output  # Compiled binaries go here

# Ensure permissions allow SSH user access
icacls "C:\Binaries" /grant "${env:USERNAME}:(OI)(CI)F"
```

## Step 5: Test Connection from Linux

```bash
# Test SSH
ssh -i ~/.ssh/win user@windows-ip "echo 'SSH works!'"

# Test MSVC (should show compiler version)
ssh -i ~/.ssh/win user@windows-ip "cl.exe"

# Test file transfer
echo "int main() { return 0; }" > /tmp/test.cpp
scp -i ~/.ssh/win /tmp/test.cpp user@windows-ip:C:/Binaries/input/
```

## Step 6: Configure halo-forge

### Option A: Config File

Create `configs/raft_windows_msvc.yaml`:

```yaml
# RAFT with MSVC Verifier for Windows training
base_model: Qwen/Qwen2.5-Coder-0.5B
num_cycles: 6
output_dir: models/raft

verifier:
  type: msvc
  host: 10.0.0.152         # Your Windows IP
  user: keys               # Your Windows username
  ssh_key: ~/.ssh/win      # Path to SSH private key
```

### Option B: CLI Arguments

```bash
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier msvc \
  --output results/windows/baseline.json
```

## Troubleshooting

### SSH Connection Refused

- Ensure sshd service is running: `Get-Service sshd`
- Check firewall: `Get-NetFirewallRule -Name 'OpenSSH*'`

### MSVC Not Found in SSH Session

- Ensure MSVC is in PowerShell profile
- Test: `ssh user@host "powershell -Command cl.exe"`
- Try: `ssh user@host "cmd /c vcvars64.bat && cl.exe"`

### Permission Denied on Binaries Directory

- Check directory permissions: `icacls C:\Binaries`
- Ensure SSH user has write access

### Slow Compilation

Disable Windows Defender real-time scanning for `C:\Binaries`:

```powershell
Add-MpPreference -ExclusionPath "C:\Binaries"
```

## Security Notes

- Use key-based authentication only (disable password auth)
- Restrict SSH to specific IP addresses if possible
- Consider using a dedicated VM for testing
- The build server does not need internet access

---

## Alternative: MinGW (No Windows Needed)

If you don't have a Windows machine, use the MinGW cross-compiler:

```bash
# Install MinGW (Fedora)
sudo dnf install mingw64-gcc-c++

# Install MinGW (Ubuntu)
sudo apt install mingw-w64

# Use MinGW verifier
halo-forge benchmark run \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --prompts datasets/windows_curriculum/windows_systems_full_rlvr.jsonl \
  --verifier mingw \
  --output results/windows/baseline_mingw.json
```

**Note**: MinGW can only verify compilation, not execution. For full verification (compile + run + output check), use MSVC.
