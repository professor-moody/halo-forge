# Contributing to Halo Forge

Thank you for your interest in contributing to Halo Forge! This document outlines the process for contributing to this project.

## Developer Certificate of Origin (DCO)

This project uses a DCO to ensure that contributors have the right to submit their contributions. By contributing to this project, you certify that you have the right to submit the work under the project's open source license.

### What is the DCO?

The DCO is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing. The full text of the DCO is available at [developercertificate.org](https://developercertificate.org/) and is reproduced below:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to Sign Off Your Commits

You must sign off on each commit to certify that you agree to the DCO. This is done by adding a `Signed-off-by` line to your commit message.

**Using the `-s` flag (recommended):**

```bash
git commit -s -m "Your commit message"
```

This will automatically add a line like this to your commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

**Configuring Git (one-time setup):**

Make sure your Git user name and email are configured:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Signing off past commits:**

If you forgot to sign off on a commit, you can amend it:

```bash
git commit --amend -s
```

For multiple commits, you can use an interactive rebase:

```bash
git rebase -i HEAD~n  # where n is the number of commits
# Then mark commits as 'reword' or 'edit' and add sign-off to each
```

## How to Contribute

### Reporting Issues

- Check existing issues to avoid duplicates
- Use a clear, descriptive title
- Include steps to reproduce, expected behavior, and actual behavior
- Include system information (OS, Python version, hardware) when relevant

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, focused commits
3. **Sign off on all commits** using `git commit -s`
4. **Test your changes** thoroughly
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description of the changes

### Code Style

- Follow existing code patterns and style
- Include docstrings for public functions and classes
- Add comments for complex logic
- Keep commits atomic and focused

### Pull Request Checklist

- [ ] All commits are signed off (DCO)
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] PR description clearly explains the changes

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.

## License

By contributing to Halo Forge, you agree that your contributions will be licensed under the Apache License 2.0.

---

Copyright 2025 Halo Forge Labs LLC
