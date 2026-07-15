"""One-shot seeder: reads JSON files from this directory and upserts rows into the DB."""

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

SEEDERS_DIR = Path(__file__).parent


def _get_database_url() -> str:
    import os

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


async def seed() -> None:
    from urgenurse.api.models import Base, User  # noqa: F401 — registers User mapper
    from urgenurse.api.services.auth_service import hash_password

    engine = create_async_engine(_get_database_url())
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]

    async with async_session() as session:
        for mapper in Base.registry.mappers:
            model = mapper.class_
            table = model.__tablename__
            json_file = SEEDERS_DIR / f"{table}.json"
            if not json_file.exists():
                continue

            records: list[dict] = json.loads(json_file.read_text())
            if not records:
                continue

            processed = []
            for record in records:
                row = dict(record)
                if table == "users" and "password" in row:
                    row["hashed_password"] = hash_password(row.pop("password"))
                processed.append(row)

            stmt = insert(model).values(processed).on_conflict_do_nothing()
            await session.execute(stmt)

        await session.commit()

    await engine.dispose()
    print("Seeding complete.")


if __name__ == "__main__":
    try:
        asyncio.run(seed())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
