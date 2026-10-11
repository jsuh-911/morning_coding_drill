# The drill as a web app

The question on the left, a sandbox on the right, two buttons. Nothing under
`morning_drill/` is modified: the verdict, the constraints, the hint ladder
and the progress decision are the same code the notebook runs, and both write
the same `drill_records.json`.

```bash
./drill      # http://localhost:8000
```

## Run and Submit are different

**Run** executes your code and shows what it produced. It costs nothing: no
attempt is counted, the record is untouched, the hint ladder does not move.
This is where the pandas actually gets worked out.

**Submit** counts an attempt. Code that raises, or that never binds `answer`,
is turned away *without* counting one — `Session.submit()` has nowhere to put
a traceback, so a crashed run would otherwise be recorded as a wrong answer
with no explanation of what went wrong.

Bind your result to `answer`. `df`, `pd` and `np` are already in scope.

| | |
|---|---|
| `⌘↩` | Run |
| `⇧⌘↩` | Submit |
| `esc` | close a sheet |

**Flag** is for a question that is wrong: it asks for one thing and checks
another, or leaves something out that you cannot know. It counts for nothing,
any attempts already spent on it are taken back out of the progress counts and
the scores, it is never dealt again (the record keeps the flag, so tomorrow
too), and a new question takes its place. The reason you type is kept in
`drill_records.json` and `logs/session.jsonl`.

A miss keeps your code in the editor — that is the point of the hint ladder.
A new challenge starts clean. Drafts survive a page reload.

## Configuration

Every knob is an environment variable, so the same image runs on localhost and
on a host without a code change.

| variable | default | what it does |
|---|---|---|
| `DRILL_DATA_DIR` | the project root | where `progress.json`, `drill_records.json` and `logs/` live |
| `DRILL_EXEC_MODE` | `subprocess` | `subprocess` or `inprocess` |
| `DRILL_EXEC_TIMEOUT` | `15` | seconds before a sandboxed run is killed |
| `DRILL_EXEC_MEMORY_MB` | `2048` | address-space cap for a sandboxed run |
| `DRILL_AUTH_TOKEN` | unset | when set, every `/api/*` call must present it |
| `DRILL_AGENTS` | `1` | `0` runs on rules alone even with a key present |

`.env` still works for the provider key: `load_env()` uses
`override=False`, so a real environment variable always wins over the file.

## Deploying it

This app runs Python that a human typed. How much work deployment is depends
entirely on who can reach it.

### localhost

Nothing to do. The default `DRILL_EXEC_MODE=subprocess` kills a run after
`DRILL_EXEC_TIMEOUT` seconds, so an accidental infinite loop shows a timeout
message instead of wedging the server. It costs about half a second per run.

If that latency bothers you, `DRILL_EXEC_MODE=inprocess` skips the pandas
re-import. It has **no timeout** -- a thread running `while True` cannot be
killed safely -- so a runaway loop hangs the server until you Ctrl-C.

### Your machine, reachable from your phone

```bash
tailscale serve 8000
```

HTTPS on a `*.ts.net` name that only your devices can resolve, with nothing
open to the internet. The process still runs on your Mac, so state, keys and
the sandbox question stay exactly as simple as localhost. Cloudflare Tunnel
with an Access policy is the same idea.

### Actually hosted

In this order, and do not skip the first two:

**1. Set `DRILL_AUTH_TOKEN`.** An unauthenticated endpoint that `exec`s its
POST body is an open shell, and separately a way to spend money on the agents.
The browser picks the token up from `?token=...` once and keeps it. Better
still, put Cloudflare Access or `oauth2-proxy` in front so authentication is
not this app's problem.

**2. Keep `DRILL_EXEC_MODE=subprocess`** (the default). Caps CPU, address space and wall
clock. Note what it does *not* do: it does not cut off the network or the
filesystem. For that, run the child in a container with `--network=none` and
a read-only mount — the protocol in `web/runner.py` is unchanged, only the
transport differs.

**3. Mount a volume and point `DRILL_DATA_DIR` at it.** A container
filesystem resets on every deploy. `drill_records.json` is the whole adaptive
mechanism — lose it and delayed retrieval, repeated-mistake targeting and
spec de-duplication all silently start over.

**4. Put the provider key in the platform's secret store,** not in the image.

```bash
docker build -t morning-drill .
docker run -p 8000:8000 \
  -v "$PWD/drill-data:/data" \
  -e ANTHROPIC_API_KEY=... \
  -e DRILL_AUTH_TOKEN="$(openssl rand -hex 24)" \
  morning-drill
```

## Where the seams are

- `web/executor.py` — `run_user_code()` is the only thing the app knows about
  execution. Swap its body for a container or a remote sandbox and nothing
  else changes.
- `web/views.py` — the one place that reads `Session._last_state`. It renders
  as data what `Session._display_challenge` and `_display_review` print as
  text. If those two ever diverge, this is the file to look at.
- `web/config.py` — every environment variable, read in one place.
