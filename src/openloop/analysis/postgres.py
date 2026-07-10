"""Postgres persistence for Phase 1 sealed-analysis inputs and artifacts."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from openloop.analysis.store import AnalysisArtifact, InputFile, InputManifest


class PostgresInputStore:
    """Durable controller-staged inputs, one manifest per analysis job."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_inputs (
                    job_id     TEXT PRIMARY KEY,
                    input_ref  TEXT NOT NULL,
                    files      JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresInputStore.setup() must be called first")
        return self._pool

    async def stage(self, manifest: InputManifest) -> None:
        pool = self._require_pool()
        files = [
            {"name": file.name, "content_b64": base64.b64encode(file.content).decode()}
            for file in manifest.files
        ]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_inputs (job_id, input_ref, files, created_at, updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (job_id) DO UPDATE SET
                    input_ref = EXCLUDED.input_ref,
                    files = EXCLUDED.files,
                    updated_at = now()
                """,
                manifest.job_id,
                manifest.input_ref,
                json.dumps(files),
                manifest.created_at,
            )

    async def get(self, job_id: str, input_ref: str) -> InputManifest | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT job_id, input_ref, files, created_at
                FROM analysis_inputs
                WHERE job_id = $1 AND input_ref = $2
                """,
                job_id,
                input_ref,
            )
        if row is None:
            return None
        files = json.loads(row["files"]) if row["files"] else []
        return InputManifest(
            job_id=row["job_id"],
            input_ref=row["input_ref"],
            files=tuple(
                InputFile(
                    name=file["name"],
                    content=base64.b64decode(file["content_b64"]),
                )
                for file in files
            ),
            created_at=row["created_at"] or datetime.now(timezone.utc),
        )


class PostgresArtifactStore:
    """Durable report artifacts, overwritten idempotently by job identity."""

    _REF_PREFIX = "analysis://"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    @classmethod
    def ref_for(cls, job_id: str) -> str:
        return f"{cls._REF_PREFIX}{job_id}/report.md"

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_artifacts (
                    job_id       TEXT PRIMARY KEY,
                    artifact_ref TEXT NOT NULL UNIQUE,
                    body         BYTEA NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresArtifactStore.setup() must be called first")
        return self._pool

    async def put(self, job_id: str, body: bytes) -> str:
        pool = self._require_pool()
        ref = self.ref_for(job_id)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_artifacts (job_id, artifact_ref, body, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (job_id) DO UPDATE SET
                    artifact_ref = EXCLUDED.artifact_ref,
                    body = EXCLUDED.body,
                    updated_at = now()
                """,
                job_id,
                ref,
                body,
            )
        return ref

    async def get(self, artifact_ref: str) -> AnalysisArtifact | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT job_id, artifact_ref, body, created_at
                FROM analysis_artifacts
                WHERE artifact_ref = $1
                """,
                artifact_ref,
            )
        if row is None:
            return None
        return AnalysisArtifact(
            job_id=row["job_id"],
            artifact_ref=row["artifact_ref"],
            body=bytes(row["body"]),
            created_at=row["created_at"] or datetime.now(timezone.utc),
        )
