# 0001 — Surface message standards

- **Status:** Accepted
- **Date:** 2026-07-16

## Context

User-visible strings had grown ad hoc alongside the features that post them.
The confirmation card said "OpenHands needs confirmation", leaking the vendor
backend; the first fix ("Coding worker needs confirmation") still leaked our
own internal component name. Refusal replies mixed styles ("⛔ approvals are
not available" vs "⛔ That decision is stale or already resolved."). Nothing
recorded which emoji meant what, or why the approval card says Approve/Deny
while the confirmation card says Accept/Reject.

Users reason about *their request*, not our topology. Component names in
surface text leak architecture, break silently when a backend is swapped, and
read as jargon. Without a written standard, every new surface string re-makes
these choices and drifts.

## Decision

Every string a user can see on a surface (today: Slack — cards, fallback
notification text, button replies, assistant-thread status, final
deliverables) follows the standards below. Logs, exceptions, docstrings,
metric names, and action ids are internal and out of scope — they may freely
name components (OpenHands, coding worker, engine, …).

### The subject rule

**Never name an internal component or backend in user-visible text.** Not the
vendor ("OpenHands") and not our own architecture ("coding worker",
"workflow", "engine", "sandbox").

Pick the subject in this order:

1. **No subject** — describe the state or the action:
   `Confirmation needed: {summary}`, `Decision recorded; resuming work.`
2. **"task"** — when a subject is grammatically required, use the thing the
   user asked for: `This task can't be resumed right now.`,
   `Only the user who approved this task may decide.`
3. **The app itself** — only via Slack's own prefixing (see Mechanics); we
   never write the app name into a string ourselves.

The `{summary}` interpolated into cards comes from the model/worker
(`pending_action_summary`, approval `req.summary`). Prompting and validation
for those live upstream, but the same rule applies: a summary that names
internals is a bug at its source, not something the card template should
launder.

### Emoji vocabulary

One emoji per message, always leading, drawn from this closed set. The
🚫-vs-⛔ distinction is load-bearing: it tells the user *who* said no.

| Emoji | Meaning | Used in |
|---|---|---|
| ⏳ | Waiting on a human **before** work starts (approval card) | `approval_blocks` |
| ⏸️ | Work is **paused mid-run** on a human decision (confirmation card) | `openhands_decision_blocks` |
| ✅ | A human said yes / a decision was durably recorded | `resolution_message`, decision replies |
| 🚫 | A **human** denied it | `resolution_message` |
| ⛔ | The **system** refused it (stale, unauthorized, unavailable, invalid state) | resolver guards |
| ⚠️ | Something broke; transient; user may retry | Slack error reply |

Don't invent new emoji per message. If a new state genuinely doesn't fit,
extend this table in the same change that introduces the string.

### Message classes and templates

**Cards (Block Kit with buttons).** Headline is
`{emoji} *{Label}:* {summary}` — bold label, colon, summary. Label is a state
noun phrase, ≤3 words.

- Approval (pre-work): `⏳ *Approval required:* {summary}` + italic approver
  line, **Approve** / **Deny** buttons (`style: primary` / `danger`).
- Confirmation (mid-run): `⏸️ *Confirmation needed:* {summary}`,
  **Accept** / **Reject** buttons (`style: primary` / `danger`).

The two verb pairs are deliberate and must not be mixed: **Approve/Deny**
gates work that hasn't started; **Accept/Reject** gates a specific action of
work already running.

**Fallback `text=` (notification preview).** Every Block Kit post carries a
plain-text `text=` mirroring the headline **without markdown or emoji**:
`Confirmation needed: {summary}`. This is what push notifications and screen
readers get.

**Resolutions and button replies.** Complete sentences, capitalized, terminal
period. Name the human actor when one acted:

- `✅ Approved by {approver} — {detail}`
- `🚫 Denied by {approver}.`
- `Action accepted by @{actor}; resuming…` (card collapse after a decision)
- `✅ Decision recorded; resuming work.`

System refusals state what happened in the user's terms and, when useful,
what to do — never internal state names:

- `⛔ That decision is stale or already resolved.`
- `⛔ Only the user who approved this task may decide.`
- `⛔ This task can't be resumed right now.`
- `⚠️ Something went wrong starting that. Please try again.`

**Progress status (assistant-thread status).** Subjectless
present-continuous phrase with a trailing `…` (the real ellipsis character):
`is working on the changes…`, `is running the sealed analysis…`. Slack
prepends the app's display name, so the app is the subject — which is exactly
the abstraction we want. Phrases describe the *milestone the user waits on*,
not the internal step name (`branch` → "is working on the changes…").

**Final deliverables.** Prose/markdown through the deliverable path
(`markdown_text=`). Normal writing; no emoji prefix; degradations are stated
inline in italics (`_(The full report `{ref}` could not be retrieved.)_`).

### Mechanics

- **Ellipsis** is `…` (one character), used only for in-progress states.
- **Em-dash** `—` separates actor from detail in resolutions.
- **Sentences** end with a period; card headlines and status phrases don't.
- **Actors** render as `@{handle}` or `{approver}` exactly as resolved from
  the surface — never a raw user id.
- Buttons: affirmative is always `style: primary`, destructive/negative is
  `style: danger`; labels are single verbs.

### Where the strings live

User-visible strings are confined to three files — new ones go there too, so
this record stays reviewable against a small surface:

- `src/openloop/surfaces/approvals.py` — cards and resolution lines
  (surface-agnostic, unit-testable without Slack).
- `src/openloop/sessions/delivery.py` — fallback `text=`, deliverable posting.
- `src/openloop/sessions/runner.py` — button-reply/refusal strings, progress
  default; workflow phrase generators (`_worker_phase`, `_analysis_phase`)
  feed it from `src/openloop/workflows/*.py`.

`src/openloop/surfaces/slack.py` holds only the greeting and the generic ⚠️
error; it must not grow message copy — it wires Bolt to the pieces above.

### Checklist for a new user-visible string

1. No internal component, backend, or step name — subject is the task or
   nothing.
2. Emoji from the table, leading, at most one — and 🚫 vs ⛔ matches who
   refused.
3. Card headline pattern or complete sentence, per class.
4. Block Kit posts carry a plain fallback `text=`.
5. Lives in one of the files above, near its siblings.
6. Tests pin structure (action ids, values), not copy — wording stays
   editable without test churn.

## Consequences

- Backends and internal components can be renamed or swapped without any
  user-visible change; nothing in Slack promises an implementation.
- New surfaces (beyond Slack) inherit a ready-made voice: the message classes
  are surface-agnostic, only the rendering (Block Kit, status API) is
  Slack-specific.
- Copy edits stay cheap: tests deliberately pin structure, not wording.
- The closed emoji set and confined string locations make drift visible in
  review — a new emoji or a string in the wrong file is a flag, not a style
  preference.
- Summaries generated by the model/worker are the remaining leak vector; the
  rule pushes fixes upstream to prompting/validation rather than into card
  templates.
