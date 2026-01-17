# Copyright Placement Guide for Halo Forge

This document outlines where to place copyright notices throughout the halo-forge project.

**Standard Copyright Notice:**
```
Copyright 2025 Halo Forge Labs LLC
```

**With License (for headers):**
```
Copyright 2025 Halo Forge Labs LLC
SPDX-License-Identifier: Apache-2.0
```

---

## 1. LICENSE File

Place at the very top, before the Apache 2.0 license text.

```
Copyright 2025 Halo Forge Labs LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 2. Source Code File Headers

Add to the top of every source file (.py, .js, .ts, .yaml, etc.).

**Python files:**
```python
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0

"""Module docstring here."""
```

**JavaScript/TypeScript:**
```javascript
// Copyright 2025 Halo Forge Labs LLC
// SPDX-License-Identifier: Apache-2.0
```

**YAML/Shell:**
```yaml
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0
```

**HTML:**
```html
<!--
  Copyright 2025 Halo Forge Labs LLC
  SPDX-License-Identifier: Apache-2.0
-->
```

**CSS:**
```css
/*
 * Copyright 2025 Halo Forge Labs LLC
 * SPDX-License-Identifier: Apache-2.0
 */
```

---

## 3. README.md

Add a license section near the bottom.

```markdown
## License

Copyright 2025 Halo Forge Labs LLC

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
```

---

## 4. Package Configuration Files

**pyproject.toml:**
```toml
[project]
name = "halo-forge"
license = {text = "Apache-2.0"}
authors = [
    {name = "Halo Forge Labs LLC"}
]

[project.urls]
Homepage = "https://halo-forge.io"
Repository = "https://github.com/halo-forge/halo-forge"
```

**package.json (if applicable):**
```json
{
  "name": "halo-forge",
  "license": "Apache-2.0",
  "author": "Halo Forge Labs LLC"
}
```

**setup.py (legacy, if used):**
```python
setup(
    name="halo-forge",
    author="Halo Forge Labs LLC",
    license="Apache-2.0",
)
```

---

## 5. Documentation

**docs/index.md or main doc page:**
```markdown
---
Copyright 2025 Halo Forge Labs LLC. Licensed under Apache 2.0.
---
```

**Documentation site footer (if using MkDocs, Sphinx, etc.):**

In mkdocs.yml:
```yaml
copyright: Copyright 2025 Halo Forge Labs LLC
```

In Sphinx conf.py:
```python
copyright = '2025, Halo Forge Labs LLC'
```

---

## 6. Website (halo-forge.io)

**Footer:**
```
© 2025 Halo Forge Labs LLC. All rights reserved.
Halo Forge is a registered trademark of Halo Forge Labs LLC.
```

**Note:** Website content itself is typically "All rights reserved" (not Apache licensed), while the software is Apache 2.0.

---

## 7. Docker/Container Files

**Dockerfile:**
```dockerfile
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim
...
```

**docker-compose.yml:**
```yaml
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0

services:
  ...
```

---

## 8. CI/CD Configuration

**GitHub Actions (.github/workflows/*.yml):**
```yaml
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0

name: CI
on: [push, pull_request]
...
```

---

## 9. Configuration Templates

Any example configs, templates, or sample files that ship with the project:

```yaml
# Copyright 2025 Halo Forge Labs LLC
# SPDX-License-Identifier: Apache-2.0
#
# Example configuration file for Halo Forge
```

---

## 10. NOTICE File (Optional but Recommended for Apache 2.0)

Create a NOTICE file in the repo root for attribution:

```
Halo Forge
Copyright 2025 Halo Forge Labs LLC

This product includes software developed by Halo Forge Labs LLC.
https://halo-forge.io

Third-party licenses:
[List any third-party dependencies and their licenses here]
```

---

## Files to Skip

Don't add copyright headers to:

- `.gitignore`
- `.env.example` (keep minimal)
- `requirements.txt` / `requirements-dev.txt`
- Generated files (build outputs, compiled assets)
- Third-party vendored code (keep their original headers)
- Data files (JSON, CSV) unless they're templates

---

## Automation Tips

**Add headers automatically with a pre-commit hook:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.5.4
    hooks:
      - id: insert-license
        files: \.py$
        args:
          - --license-filepath
          - .license-header.txt
```

**.license-header.txt:**
```
Copyright 2025 Halo Forge Labs LLC
SPDX-License-Identifier: Apache-2.0
```

---

## Summary Checklist

| Location | Priority | Status |
|----------|----------|--------|
| LICENSE file | Required | ☐ |
| README.md | Required | ☐ |
| Source code headers (.py) | Required | ☐ |
| pyproject.toml | Required | ☐ |
| NOTICE file | Recommended | ☐ |
| CONTRIBUTING.md | Recommended | ☐ |
| Documentation | Recommended | ☐ |
| Website footer | Recommended | ☐ |
| Dockerfile | Nice to have | ☐ |
| CI/CD configs | Nice to have | ☐ |
| Config templates | Nice to have | ☐ |

---

Copyright 2025 Halo Forge Labs LLC
