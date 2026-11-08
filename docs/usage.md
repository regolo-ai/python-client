# Session history, usage and billing

| SDK method                                      | Scope/defaults                                                                 |
|-------------------------------------------------|--------------------------------------------------------------------------------|
| `get_loaded_models()`                           | Open sessions, including pending rows with null `load_time`                    |
| `get_runtime_status(id)`                        | One owned current/historical session; final failure/stop reason                |
| `get_runtime_billing(id)`                       | Lifetime billing through unload or observation time                            |
| `get_user_inference_status(...)`                | Windowed billing/status; defaults to current UTC month; requires `model:write` |
| `get_inference_sessions(...)`                   | One cursor page; no dates means **all history**                                |
| `iter_inference_sessions(...)`                  | Lazy traversal of every matching history page                                  |
| `get_usage_summary(...)`                        | All-page aggregation of server-calculated windowed usage                       |
| `export_sessions_csv(path_or_text_handle, ...)` | All matching pages; status/billing default on, errors opt-in                   |

Failed deployments close and disappear from open sessions. Recover full failure
text with history's `status="error", include_error=True`. Stored history status
is `error`; runtime/public status maps it to `failed`. Stopped sessions retain
their administrative reason.

## Filters and pagination

History accepts `limit` (1–200), `before_session_id`, `month`, `time_range_start`,
`time_range_end`, `model_name`, `status`, `include_status`, `include_error` and
`include_billing`. The include flags default off; omitted fields remain distinct
from explicit null values in SDK response dictionaries.

```python
page = client.get_inference_sessions(limit=50, month="102026", include_billing=True)
if page.get("next_cursor") is not None:
    older = client.get_inference_sessions(
        limit=50,
        month="102026",
        before_session_id=page["next_cursor"],
        include_billing=True,
    )
```

Rows are newest first by session ID. `count` describes the page; `total` counts
all matches ignoring the cursor; `next_cursor` selects the next older page.
History `total_cost` is only the **page** cost. The iterator preserves filters,
checks descending IDs/cursor consistency and fails on repeated/out-of-order rows.

`month` uses MMYYYY in UTC. It is mutually exclusive with explicit ranges, which
require both bounds as timezone-aware datetimes or ISO 8601 strings. Incomplete,
reversed and zero-length ranges are rejected. History uses an exclusive upper
boundary; the older user-status route describes its end as inclusive while
billing clips to the supplied bounds. Prefer history for cursor/window reports.

```python
summary = client.get_usage_summary(
    time_range_start="2026-10-01T00:00:00+02:00",
    time_range_end="2026-11-01T00:00:00+01:00",
)
```

## Summary semantics

The JSON-compatible result includes `currency="EUR"`, filters, `observed_at`,
`sessions`, `open_sessions`, `duration_seconds`, `replica_seconds`,
`billable_replica_hours`, `cost_euro`, `status_counts`, `by_model` and `by_gpu`.

It sums `cost_euro` over **all pages**, never `cost_euro_from_start`, using decimal
arithmetic internally and returning JSON numbers. Missing/invalid billing raises
a protocol error instead of producing false zeros.

Devmid computes replica usage from recorded observations and a snapshotted
customer rate. Wall-clock time differs from replica time; zero replicas accrue
no replica usage. Server rounding applies to integrated replica time. The SDK
does not recompute prices or round intervals. Billable replica hours sum the
already rounded per-session amounts. Duration sums can include overlapping
deployments; they are not unique user elapsed time.

Filtered `cost_euro` covers the reporting window; `cost_euro_from_start` and
runtime billing cover the lifetime. Summing lifetime costs across months
double-counts usage. These are hosting charges, separate from inference-token
billing. Costs of running sessions can change while fetching pages; keyset
pagination does not create an atomic invoice snapshot. Preserve `observed_at`.

## CSV and CLI

```python
client.export_sessions_csv("october.csv", month="102026", include_error=True)
```

The export includes IDs, model/GPU, status, optional error, timestamps, open-state,
billing/lifetime costs and effective context length. It returns the row count.
Null/omitted values become empty cells. Supplied text handles remain caller-owned.
A failed export can leave a partial file; errors propagate.

```bash
regolo inference sessions --status error --include-status --include-error --all
regolo inference usage --month 102026 --format json
regolo inference sessions --month 102026 --csv october.csv
regolo inference billing 42
```

CLI CSV exports include status/billing without additional flags. Error text can
be large and is never truncated, so request it when needed.

Devmid's public API has no inference request/token counts, account balance,
invoices, spending limits or time-series traffic metrics. Retrieve serving API
`usage` metadata through inference `full_output=True` when available; the
management summary does not infer these figures from deployment duration.

## Scaling history and complete exports

`get_loaded_models(include_scaling=True)` and
`get_inference_sessions(include_scaling=True, ...)` request bounded inline
scaling data. The same flag works through the session iterator, exact session
lookup, combined details and JSON export. Every returned field is preserved:
actual server keys, cursor/truncation indicators, live assumptions, finalization
and frozen intervals. The SDK does not infer new events from cost changes.

For the paginated detail route, supply a current public OpenAPI export or permit
lazy schema discovery, then call `get_scaling_history(session_id, limit=100,
cursor=None)`. Pass its returned `next_cursor` into the next call. The exact route
and cursor argument are selected from the authenticated public schema; an absent,
ambiguous, anonymous or Basic-auth route fails locally. `iter_scaling_history`
follows `next_cursor`, rejecting repeated/invalid cursors; select `collection`
if your response's object array is named differently from `intervals`. Use raw
pages when page-level finalization/assumption metadata is needed.

`get_inference_session(id)` uses an exact keyset lookup and rejects a neighboring
row or a missing ID. Another owner's data must still be rejected by Devmid; no
owner identity can be overridden through this client.

`export_sessions_json(path, ...)` follows all session pages and atomically writes
all nested/future fields to owner-only JSON. If a request fails, an existing
export stays intact. Enable desired optional fields explicitly. CSV retains its
fixed tabular columns; nested scaling belongs in JSON. By default CSV prefixes
formula-like strings with an apostrophe for spreadsheet safety. Use
`spreadsheet_safe=False` only when literal CSV cell values are required.
