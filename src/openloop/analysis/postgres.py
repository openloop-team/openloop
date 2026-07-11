"""Postgres persistence for Phase 1 sealed-analysis inputs and artifacts."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from openloop.analysis.store import AnalysisArtifact, AnalysisAttempt, InputFile, InputManifest


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


class PostgresAnalysisAttemptStore:
    """Durable analysis attempt accounting, separate from report artifacts."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_attempts (
                        attempt_id       TEXT PRIMARY KEY,
                        job_id           TEXT NOT NULL,
                        status           TEXT NOT NULL,
                        cost_usd         DOUBLE PRECISION,
                        prompt_tokens    INTEGER,
                        completion_tokens INTEGER,
                        error            TEXT,
                        created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                        charged_at       TIMESTAMPTZ,
                        settled_at       TIMESTAMPTZ,
                        updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS analysis_attempts_job_idx "
                    "ON analysis_attempts (job_id, created_at DESC)"
                )
        except BaseException:
            # The app may replace a setup-failed store before shutdown sees it;
            # setup therefore owns cleanup of a pool it successfully opened.
            await self.close()
            raise

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError("PostgresAnalysisAttemptStore.setup() must be called first")
        return self._pool

    async def begin(self, attempt_id: str, job_id: str) -> tuple[AnalysisAttempt, bool]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO analysis_attempts (attempt_id, job_id, status)
                VALUES ($1, $2, 'started')
                ON CONFLICT (attempt_id) DO NOTHING
                RETURNING *
                """,
                attempt_id,
                job_id,
            )
            if row is not None:
                return _row_to_attempt(row), True
            existing = await conn.fetchrow(
                "SELECT * FROM analysis_attempts WHERE attempt_id = $1", attempt_id
            )
        if existing is None:  # defensive: a deleted row raced the conflict read
            raise RuntimeError(f"analysis attempt {attempt_id} disappeared during begin")
        return _row_to_attempt(existing), False

    async def get(self, attempt_id: str) -> AnalysisAttempt | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM analysis_attempts WHERE attempt_id = $1", attempt_id
            )
        return _row_to_attempt(row) if row else None

    async def charge(
        self,
        attempt_id: str,
        *,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> AnalysisAttempt:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # A charged attempt accepts only monotonic growth: the iterative
            # strategy re-charges cumulative totals after every completion, and
            # an equal replay after a crash is safe. charged_at keeps the first
            # observation time.
            row = await conn.fetchrow(
                """
                UPDATE analysis_attempts
                SET status = 'charged', cost_usd = $2, prompt_tokens = $3,
                    completion_tokens = $4,
                    charged_at = COALESCE(charged_at, now()), updated_at = now()
                WHERE attempt_id = $1
                  AND (status = 'started'
                       OR (status = 'charged'
                           AND cost_usd <= $2
                           AND prompt_tokens <= $3
                           AND completion_tokens <= $4))
                RETURNING *
                """,
                attempt_id,
                cost_usd,
                prompt_tokens,
                completion_tokens,
            )
            if row is not None:
                return _row_to_attempt(row)
            existing = await conn.fetchrow(
                "SELECT * FROM analysis_attempts WHERE attempt_id = $1", attempt_id
            )
        attempt = _require_attempt(existing, attempt_id)
        if attempt.status == "settled":
            _assert_same_charge(attempt, cost_usd, prompt_tokens, completion_tokens)
            return attempt
        if attempt.status == "charged":
            # The UPDATE matched neither started nor monotonic-charged, so this
            # charge would decrease a cumulative total — fail loudly.
            raise RuntimeError(
                f"analysis attempt {attempt_id} already has different "
                "charge data: cumulative charge would decrease"
            )
        raise RuntimeError(
            f"analysis attempt {attempt_id} is {attempt.status}; cannot charge"
        )

    async def settle(self, attempt_id: str) -> AnalysisAttempt:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE analysis_attempts
                SET status = 'settled', settled_at = now(), updated_at = now()
                WHERE attempt_id = $1 AND status = 'charged'
                RETURNING *
                """,
                attempt_id,
            )
            if row is not None:
                return _row_to_attempt(row)
            existing = await conn.fetchrow(
                "SELECT * FROM analysis_attempts WHERE attempt_id = $1", attempt_id
            )
        attempt = _require_attempt(existing, attempt_id)
        if attempt.status == "settled":
            return attempt
        raise RuntimeError(f"analysis attempt {attempt_id} is {attempt.status}; cannot settle")

    async def mark_unknown(self, attempt_id: str, error: str) -> AnalysisAttempt:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE analysis_attempts
                SET status = 'unknown', error = $2, updated_at = now()
                WHERE attempt_id = $1 AND status != 'settled'
                RETURNING *
                """,
                attempt_id,
                error,
            )
            if row is not None:
                return _row_to_attempt(row)
            existing = await conn.fetchrow(
                "SELECT * FROM analysis_attempts WHERE attempt_id = $1", attempt_id
            )
        return _require_attempt(existing, attempt_id)


def _row_to_attempt(row) -> AnalysisAttempt:
    now = datetime.now(timezone.utc)
    return AnalysisAttempt(
        attempt_id=row["attempt_id"],
        job_id=row["job_id"],
        status=row["status"],
        cost_usd=row["cost_usd"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        error=row["error"],
        created_at=row["created_at"] or now,
        charged_at=row["charged_at"],
        settled_at=row["settled_at"],
        updated_at=row["updated_at"] or now,
    )


def _require_attempt(row, attempt_id: str) -> AnalysisAttempt:
    if row is None:
        raise KeyError(f"unknown analysis attempt {attempt_id}")
    return _row_to_attempt(row)


def _assert_same_charge(
    attempt: AnalysisAttempt,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if (
        attempt.cost_usd != cost_usd
        or attempt.prompt_tokens != prompt_tokens
        or attempt.completion_tokens != completion_tokens
    ):
        raise RuntimeError(
            f"analysis attempt {attempt.attempt_id} already has different charge data"
        )
