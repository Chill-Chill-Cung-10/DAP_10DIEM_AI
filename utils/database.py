import os
from typing import Any, Optional

import asyncpg
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

_postgres_pool: Optional[asyncpg.Pool] = None
_redis_client: Optional[redis.Redis] = None


async def get_postgre_client() -> asyncpg.Pool:
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")

    global _postgres_pool
    if _postgres_pool is None:
        _postgres_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
        )
    return _postgres_pool


async def get_redis_client() -> redis.Redis:
    if not REDIS_URL:
        raise ValueError("REDIS_URL is not set")

    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def fetch_query(query: str, *args: Any) -> Optional[asyncpg.Record]:
    pool = await get_postgre_client()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_all(query: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_postgre_client()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def check_health() -> dict[str, Any]:
    status: dict[str, Any] = {"postgres": False, "redis": False}

    try:
        pool = await get_postgre_client()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        status["postgres"] = True
    except Exception as exc:  # noqa: BLE001
        status["postgres_error"] = str(exc)

    try:
        client = await get_redis_client()
        status["redis"] = bool(await client.ping())
    except Exception as exc:  # noqa: BLE001
        status["redis_error"] = str(exc)

    return status
