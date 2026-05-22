"""Project-scoped MongoDB client — CRM tool'lari uchun.

Backend har project'ga alohida MongoDB database yaratadi:
    db_name = f"{prefix}{project_id}"   # masalan: project_42

Collections: built-in `users`, `chat` + custom (CollectionModel orqali).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ..config import get_settings


class MongoClient:
    def __init__(self, base_url: str, prefix: str = "project_") -> None:
        self._client = AsyncIOMotorClient(base_url)
        self._prefix = prefix

    def db(self, project_id: str | int) -> AsyncIOMotorDatabase:
        return self._client[f"{self._prefix}{project_id}"]

    async def list_collections(self, project_id: str | int) -> list[str]:
        return await self.db(project_id).list_collection_names()

    async def find(
        self,
        project_id: str | int,
        collection: str,
        *,
        filter: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 50,
        skip: int = 0,
        projection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.db(project_id)[collection].find(filter or {}, projection=projection)
        if sort:
            cursor = cursor.sort(sort)
        cursor = cursor.skip(skip).limit(limit)
        return [self._normalize(doc) async for doc in cursor]

    async def find_one(
        self,
        project_id: str | int,
        collection: str,
        *,
        filter: dict[str, Any],
    ) -> dict[str, Any] | None:
        doc = await self.db(project_id)[collection].find_one(filter)
        return self._normalize(doc) if doc else None

    async def count(
        self,
        project_id: str | int,
        collection: str,
        *,
        filter: dict[str, Any] | None = None,
    ) -> int:
        return await self.db(project_id)[collection].count_documents(filter or {})

    async def aggregate(
        self,
        project_id: str | int,
        collection: str,
        pipeline: list[dict[str, Any]],
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async for doc in self.db(project_id)[collection].aggregate(pipeline):
            out.append(self._normalize(doc))
            if len(out) >= limit:
                break
        return out

    async def insert(
        self,
        project_id: str | int,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        res = await self.db(project_id)[collection].insert_one(document)
        return str(res.inserted_id)

    async def update(
        self,
        project_id: str | int,
        collection: str,
        *,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
        many: bool = False,
    ) -> int:
        coll = self.db(project_id)[collection]
        # Ensure operator format
        ops = update if any(k.startswith("$") for k in update) else {"$set": update}
        if many:
            res = await coll.update_many(filter, ops, upsert=upsert)
        else:
            res = await coll.update_one(filter, ops, upsert=upsert)
        return res.modified_count

    async def delete(
        self,
        project_id: str | int,
        collection: str,
        *,
        filter: dict[str, Any],
        many: bool = False,
    ) -> int:
        coll = self.db(project_id)[collection]
        if many:
            res = await coll.delete_many(filter)
        else:
            res = await coll.delete_one(filter)
        return res.deleted_count

    @staticmethod
    def _normalize(doc: dict[str, Any] | None) -> dict[str, Any]:
        if not doc:
            return {}
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def close(self) -> None:
        self._client.close()


@lru_cache(maxsize=2)
def get_mongo_client(_settings_id: int = 0) -> MongoClient:
    s = get_settings()
    return MongoClient(s.mongo_base_url, prefix=s.mongo_db_prefix)
