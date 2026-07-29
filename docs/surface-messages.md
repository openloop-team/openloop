# Surface message standards

The governing decision is
[ADR 0001](adr/0001-surface-message-standards.md): never name an internal
component or backend in user-visible text. This document carries the concrete
vocabulary, templates, and locations that follow from that rule, and is expected
to change as message classes and surfaces are added.

## Emoji vocabulary

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

## Message classes and templates

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
`is working on the changes…`. Slack prepends the app's display name, so the app
is the subject — which is exactly the abstraction we want. Phrases describe the
*milestone the user waits on*, not the internal step name (`branch` → "is
working on the changes…").

**Final deliverables.** Prose/markdown through the deliverable path
(`markdown_text=`). Normal writing; no emoji prefix; degradations are stated
inline in italics (`_(The full report `{ref}` could not be retrieved.)_`).

## Mechanics

- **Ellipsis** is `…` (one character), used only for in-progress states.
- **Em-dash** `—` separates actor from detail in resolutions.
- **Sentences** end with a period; card headlines and status phrases don't.
- **Actors** render as `@{handle}` or `{approver}` exactly as resolved from
  the surface — never a raw user id.
- Buttons: affirmative is always `style: primary`, destructive/negative is
  `style: danger`; labels are single verbs.

## Where the strings live

User-visible strings are confined to three files — new ones go there too, so
this standard stays reviewable against a small surface:

- `src/openloop/surfaces/approvals.py` — cards and resolution lines
  (surface-agnostic, unit-testable without Slack).
- `src/openloop/sessions/delivery.py` — fallback `text=`, deliverable posting.
- `src/openloop/sessions/runner.py` — button-reply/refusal strings, progress
  default; workflow phrase generators feed it from
  `src/openloop/workflows/*.py`.

`src/openloop/surfaces/slack.py` holds only the greeting and the generic ⚠️
error; it must not grow message copy — it wires Bolt to the pieces above.

## Checklist for a new user-visible string

1. No internal component, backend, or step name — subject is the task or
   nothing.
2. Emoji from the table, leading, at most one — and 🚫 vs ⛔ matches who
   refused.
3. Card headline pattern or complete sentence, per class.
4. Block Kit posts carry a plain fallback `text=`.
5. Lives in one of the files above, near its siblings.
6. Tests pin structure (action ids, values), not copy — wording stays
   editable without test churn.
