"""Allow ``python -m legacy_csl`` execution.

``python -m legacy_csl`` による実行を許可する。
"""

from .cli import main

raise SystemExit(main())

