"""Drop-in blueprint extensions.

Each module in this package is auto-imported by :mod:`agent.flow.blueprint` and
registers its own ``BlueprintDefinition``. This directory can grow to hundreds
of independent business patterns without expanding a central switch statement.
"""
