import asyncio

from utils.database import (
    check_health,
    fetch_query,
    get_postgre_client,
    get_redis_client,
)


async def main() -> None:
    health = await check_health()
    print("health:", health)

    row = await fetch_query("SELECT 1 AS ok")
    print("postgres test:", dict(row) if row else None)

    redis_client = await get_redis_client()
    await redis_client.set("ping", "pong", ex=30)
    value = await redis_client.get("ping")
    print("redis test:", value)

    pool = await get_postgre_client()
    await pool.close()
    await redis_client.close()
    await redis_client.connection_pool.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
