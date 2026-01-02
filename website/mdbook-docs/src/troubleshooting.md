# Troubleshooting

Common issues and solutions.

## GPU Issues

### GPU Not Detected

```bash
# Check device access
ls -l /dev/dri /dev/kfd

# Add udev rules
sudo tee /etc/udev/rules.d/99-amd-kfd.rules >/dev/null <<'EOF'
SUBSYSTEM=="kfd", GROUP="render", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="card[0-9]*", GROUP="render", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### GPU Hang During Training

**Cause**: `dataloader_num_workers > 0`

**Fix**: Set in config:
```yaml
training:
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

### Low GPU Utilization

```bash
# Check utilization (requires sudo)
sudo rocm-smi --showuse

# Set environment variable
export HSA_ENABLE_SDMA=0
```

## Memory Issues

### Out of Memory

Unlikely with 128GB, but if it happens:

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Reduce `max_seq_length`

## Training Issues

### Loss Not Decreasing

- Learning rate too high → try `1e-5`
- Data quality issues → verify training examples
- Too few epochs → increase `num_train_epochs`

### RAFT Performance Degrading

After cycle 5-6, performance may drop:

- Stop at peak performance cycle
- Use learning rate decay
- Increase prompt diversity

## Verifier Issues

### GCC Not Found

```bash
# Inside toolbox
which g++

# If missing
sudo dnf install gcc-c++
```

### Remote MSVC Connection Failed

1. Check SSH key authentication
2. Verify Windows OpenSSH is running
3. Test connection: `ssh user@host "echo test"`

## Import Errors

Make sure you're inside the toolbox:

```bash
toolbox enter halo-forge
```

## Getting Help

- GitHub Issues: [github.com/professor-moody/halo-forge/issues](https://github.com/professor-moody/halo-forge/issues)
- Check existing issues before creating new ones
- Include: error message, config, hardware info
