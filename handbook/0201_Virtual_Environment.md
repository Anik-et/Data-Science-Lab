## Why virtual environments?

Different projects need different package versions.

Example

Project A

pandas 1.5

↓

Project B

pandas 2.2

Without environments they conflict.

---

Notebook and terminal can accidentally use different interpreters.

Always verify

python -c "import sys; print(sys.executable)"

and

import sys
print(sys.executable)

inside notebook.