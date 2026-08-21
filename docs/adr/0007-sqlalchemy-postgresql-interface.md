# 0007 — Make SQLAlchemy the only PostgreSQL interface

- **Status:** Proposed
- **Date:** 2026-08-16
- **Implementation:** PostgreSQL access and statement construction in project
  code; the schemas, what the statements mean, result-to-domain mapping, and
  durable state held outside PostgreSQL otherwise unchanged.

## Context

PostgreSQL-backed stores and coordination code issue SQL text directly through
a database driver. Statements use the driver's positional parameters. Table
and column names therefore exist only inside string literals: a renamed column
is found by searching text rather than by following a reference, and parameter
numbering is bookkeeping the author maintains by hand.

Not all durable state is relational. Workspace artifacts and the conversation
state carried by their own drivers are durable, are not held in these schemas,
and no query-building layer applies to them; they are outside this record.

This is a decision about the level at which PostgreSQL access code is written,
not a portability argument — the schemas use JSONB operators, common table
expressions, and conditional writes that return the affected row deliberately,
and coordination uses session-level advisory locks. Nothing here proposes
giving those constructs up. The payoff is uneven: single-table reads and
writes express more clearly in a query-building layer, while a paginated
recovery sweep or a JSONB document merged into a column often expresses less
clearly there than in the SQL it would replace.

Two alternatives were rejected. A project-local typing layer over SQL text
avoids the dependency, but it makes the project responsible for a persistence
abstraction as well as for the product, and one nobody would understand
without reading it. An object-relational mapping goes further, replacing
statements with mapped objects; its benefits rest on a graph of related
records these tables do not form, and it is not decided here.

## Decision

All project code that accesses PostgreSQL — including stores, migrations,
health probes, and advisory-lock coordination — does so through SQLAlchemy.

- Only SQLAlchemy's dialect reaches the database driver directly. Project code
  neither imports nor calls the driver.
- Statements are written in the expression language. SQL text runs through the
  same interface and is kept only where the expression language would obscure
  what a statement does; keeping it requires a stated reason at the statement.
- No PostgreSQL access path is permanently exempt. The repository does not
  satisfy this record until every one of them is converted, and that gap must
  close.
- Metadata declared for statement construction describes a schema and never
  defines one. It emits no data definition, is never consulted to create or
  alter a schema, and where it and the database disagree the database is
  right. Building a statement never requires a reachable database.
- Values reach the database as bound parameters in both forms. No statement is
  assembled by interpolating a value into its text.
- A guarded transition remains a single conditional statement in whichever
  form carries it. Where a store's correctness rests on a condition the
  database evaluates at write time, that condition stays in the statement
  performing the write.
- Dialect portability is not a goal, and constructs particular to Postgres are
  not to be replaced in pursuit of it.

## Consequences

- Every PostgreSQL access path is converted, including stores that read
  perfectly well today and coordination that already has the connection
  semantics it needs. Those paths gain nothing individually. The gain is that
  only one route to PostgreSQL exists, and partial adoption cannot deliver it.
  The order and duration of the work are planned outside this repository; this
  record fixes only that it finishes.
- Session-level advisory locks retain one dedicated connection for the lifetime
  of each lock. Moving that connection behind SQLAlchemy does not turn the lock
  into transaction-scoped coordination or permit the connection to return to
  the pool while the lock is held.
- Completion has two conditions and only one can be checked mechanically: no
  project code imports or calls a database driver, and every SQL statement kept
  as text carries its reason. The second is settled in review. Meeting the
  first alone finishes nothing — a repository can route everything through the
  new interface and still be written entirely in unexplained text.
- Intricate statements convert at their connection layer without being
  rewritten, so a recovery sweep or a JSONB merge kept as text still satisfies
  this record. What changes is that keeping the text now costs a reason
  written at the statement, where before it was the silent default.
- Column names are still not statically checked. Ordinary metadata is looked
  up by name at run time, so a misspelled column surfaces when the statement
  is built, not from a type checker. The gain is that a name is defined once
  and referred to everywhere else, not that mistakes are caught sooner. A
  typed declaration form that would check names exists in a pre-release
  series; because this record requires the expression language, that checking
  would reach nearly every statement, so the case for taking it strengthens as
  conversion proceeds. A stable release removes the objection to depending on
  it, but not the limitation, which holds until this project adopts the typed
  form.
- Metadata restates in Python a shape the schema already defines, and the two
  can drift. The constraints above bound what that drift costs but do not
  prevent it, and a statement built from stale metadata fails when it runs.
- No performance claim is made or implied. Statement construction moves from
  the driver to a layer above it, and the effect on throughput is not a reason
  for this decision and has not been measured.
- Reintroducing direct driver access or value interpolation into statement
  text, or removing a Postgres-specific construct for the sake of neutrality,
  requires an ADR superseding this one.
