"""Postgres persistence for sealed-analysis inputs, uploads, and artifacts."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone

from openloop.analysis.store import AnalysisArtifact, AnalysisAttempt, InputFile, InputManifest
from openloop.analysis.uploads import UploadRecord
from openloop.postgres import BorrowedPostgresStore


class PostgresInputStore(BorrowedPostgresStore):
    """Durable operator-staged inputs, one manifest per capability ref."""

    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
            # Phase 4 rekeyed staging on the capability ref. Staged inputs are
            # short-lived operator artifacts, so the job-keyed Phase 1 table
            # is dropped, not migrated — anything in it must be re-staged.
            await conn.execute("DROP TABLE IF EXISTS analysis_inputs")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_staged_inputs (
                    input_ref  TEXT PRIMARY KEY,
                    files      JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    async def stage(self, manifest: InputManifest) -> None:
        pool = self._require_pool()
        files = [
            {"name": file.name, "content_b64": base64.b64encode(file.content).decode()}
            for file in manifest.files
        ]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_staged_inputs (input_ref, files, created_at, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (input_ref) DO UPDATE SET
                    files = EXCLUDED.files,
                    updated_at = now()
                """,
                manifest.input_ref,
                json.dumps(files),
                manifest.created_at,
            )

    async def get(self, input_ref: str) -> InputManifest | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT input_ref, files, created_at
                FROM analysis_staged_inputs
                WHERE input_ref = $1
                """,
                input_ref,
            )
        if row is None:
            return None
        files = json.loads(row["files"]) if row["files"] else []
        return InputManifest(
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


class PostgresUploadStore(BorrowedPostgresStore):
    """Durable surface-upload metadata (never bytes — staging is lazy)."""

    async def setup(self, pool) -> None:
        try:
            async with self._setup_connection(pool) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_uploads (
                        upload_ref TEXT PRIMARY KEY,
                        scope_key  TEXT NOT NULL,
                        name       TEXT NOT NULL,
                        size       BIGINT NOT NULL,
                        shared_by  TEXT,
                        shared_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS analysis_uploads_scope_idx "
                    "ON analysis_uploads (scope_key, shared_at)"
                )
        except BaseException:
            await self.close()
            raise

    async def record(self, upload: UploadRecord) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # First write wins: a re-delivered file event must not move an
            # already-recorded upload into a different scope.
            await conn.execute(
                """
                INSERT INTO analysis_uploads
                    (upload_ref, scope_key, name, size, shared_by, shared_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (upload_ref) DO NOTHING
                """,
                upload.upload_ref,
                upload.scope_key,
                upload.name,
                upload.size,
                upload.user,
                upload.shared_at,
            )

    async def get(self, upload_ref: str) -> UploadRecord | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM analysis_uploads WHERE upload_ref = $1",
                upload_ref,
            )
        return _row_to_upload(row) if row else None

    async def for_scope(
        self, scope_key: str, *, limit: int = 20
    ) -> list[UploadRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM (
                    SELECT * FROM analysis_uploads
                    WHERE scope_key = $1
                    ORDER BY shared_at DESC
                    LIMIT $2
                ) recent
                ORDER BY shared_at ASC
                """,
                scope_key,
                limit,
            )
        return [_row_to_upload(r) for r in rows]


def _row_to_upload(row) -> UploadRecord:
    return UploadRecord(
        upload_ref=row["upload_ref"],
        scope_key=row["scope_key"],
        name=row["name"],
        size=row["size"],
        user=row["shared_by"],
        shared_at=row["shared_at"] or datetime.now(timezone.utc),
    )


class PostgresArtifactStore(BorrowedPostgresStore):
    """Durable report artifacts, overwritten idempotently by job identity."""

    _REF_PREFIX = "analysis://"

    @classmethod
    def ref_for(cls, job_id: str) -> str:
        return f"{cls._REF_PREFIX}{job_id}/report.md"

    async def setup(self, pool) -> None:
        async with self._setup_connection(pool) as conn:
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


class PostgresAnalysisAttemptStore(BorrowedPostgresStore):
    """Durable analysis attempt accounting, separate from report artifacts."""

    async def setup(self, pool) -> None:
        try:
            async with self._setup_connection(pool) as conn:
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
            # detach from the borrowed pool immediately.
            await self.close()
            raise

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
