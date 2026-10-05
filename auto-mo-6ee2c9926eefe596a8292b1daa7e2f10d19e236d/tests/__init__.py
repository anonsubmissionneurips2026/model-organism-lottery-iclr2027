# Marks tests as a package so the `tests.*` mypy override (relaxed typing for
# tests) actually matches — without it, mypy names these modules `test_config`
# etc. and type-checks them under full strict mode.
