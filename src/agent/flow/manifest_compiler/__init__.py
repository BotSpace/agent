"""Manifest → Pydantic schema compiler.

Built-in (va modul) node'larning data-schema'si endi frontend manifestdan
(`NodeFieldManifest`) kompilyatsiya qilinadi — qo'lda yozilgan Pydantic model
o'rniga. Bu paket field-type mapping kontrakti (`field_type_map`) va kompilyator
(`compiler`, Bosqich 2) ni o'z ichiga oladi.

Reja: /Users/ico/.claude/plans/agent-arxitikturasini-yangilash-kerak-eager-thimble.md
"""

from .field_type_map import FIELD_TYPE_MAP, FieldKind, FieldTypeSpec, known_field_types

__all__ = ["FIELD_TYPE_MAP", "FieldKind", "FieldTypeSpec", "known_field_types"]
