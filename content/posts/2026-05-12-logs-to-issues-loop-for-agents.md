---
title: "Closing the logs-to-issues loop with AI agents"
date: 2026-05-12T00:00:00Z
slug: "closing-logs-to-issues-loop-with-ai-agents"
disqus_identifier: 2026-05-12
author: martin
summary: "Generating code is the easy part of agentic engineering. The harder goal is letting the agent manage the whole development cycle, including watching how a change behaves in production once it ships. The missing link is usually feedback: logs sit on the machines the service runs on, and the agent shouldn't need a login to those machines to read them. The setup below pushes logs into Grafana Cloud Loki, then gives the agent a read-only token plus <code>logcli</code>. Every N minutes the agent queries Loki for errors and opens (or comments on) GitHub issues against the repo. The agent never has to touch a VM, and there's no extra infrastructure to babysit."
comments: true
---

Generating code is the easy part of agentic engineering. The harder goal is letting the agent manage the whole development cycle, including watching how a change behaves in production once it ships. The missing link is usually feedback: logs sit on the machines the service runs on, and the agent shouldn't need a login to those machines to read them.

The setup below pushes logs into Grafana Cloud Loki, then gives the agent a read-only token plus `logcli`. Every N minutes the agent queries Loki for errors and opens (or comments on) GitHub issues against the repo. The agent never has to touch a VM, and there's no extra infrastructure to babysit.

## Grafana Cloud

[Grafana Cloud](https://grafana.com/) has a free tier that fits this use case: 50 GB of log ingestion per month, 14 days of retention, no credit card required. Sign up at Grafana, create Loki stack, and note the Loki endpoint (something like `https://logs-prod-us-central1.grafana.net`) plus the user ID.

## Install logcli

Clone the [Loki repo](https://github.com/grafana/loki) and build the CLI:

```bash
git clone https://github.com/grafana/loki.git
cd loki
make logcli
```

The binary is built at `./cmd/logcli/logcli`. Move it onto `$PATH`:

```bash
mv ./cmd/logcli/logcli /usr/local/bin/logcli
```

## Install GitHub CLI

[GitHub CLI](https://cli.github.com/) handles the issue side. Install it:

```bash
brew install gh    # macOS
apt install gh     # Debian/Ubuntu
```

Authenticate with a token scoped to `repo` (or a fine-grained PAT with read+write on issues for the one repo). Run once:

```bash
gh auth login
```

The token lives in the keychain after that.

## Wire it into Claude Code

In Grafana Cloud, create an Access Policy scoped to `logs:read` only (no `logs:write`, no admin), then generate a token under that policy. That token is the agent's entire permission surface for Loki.

Put the connection details and the allowlist in `.claude/settings.local.json` (per-project, gitignored by default, so the token stays out of git):

```json
{
  "env": {
    "LOKI_ADDR": "https://logs-prod-us-central1.grafana.net",
    "LOKI_USERNAME": "1234567",
    "LOKI_PASSWORD": "glc_xxxxxxxxxxxxxxxxxxxxx"
  },
  "permissions": {
    "allow": [
      "Bash(logcli query *)",
      "Bash(logcli labels *)",
      "Bash(logcli series *)",
      "Bash(gh issue list *)",
      "Bash(gh issue view *)",
      "Bash(gh issue create *)",
      "Bash(gh issue comment *)"
    ]
  }
}
```

The agent's bash sessions inherit the env vars, so `logcli` connects without anything on the command line. Read-only access is enforced on two layers. The allowlist limits the agent to query commands, and the token itself has no write scope.

## Useful queries

Single service, error level only, last 30 minutes:

```bash
logcli query '{service_name="myservice", level="error"}' --since=30m --limit=200
```

Multiple services, error or fatal:

```bash
logcli query '{service_name=~"myservice|otherservice", level=~"error|fatal"}' --since=30m
```

Free-text filter on top of label selector:

```bash
logcli query '{service_name="myservice"} |= "panic"' --since=1h
```

## The agent prompt

Save as `.claude/commands/scan-logs.md`:

```text
Scan production logs for the last 30 minutes and file issues for new errors.

1. Find which services have errors in the window:
   logcli query --since=30m \
     'sum by (service_name) (count_over_time({level=~"error|fatal"}[30m]))'

2. For each service with non-zero count, pull a deduped sample:
   logcli query --since=30m --limit=1000 --output=raw \
     '{service_name="<svc>", level=~"error|fatal"} | logfmt | line_format "{{.msg}}"' \
     | sort | uniq -c | sort -rn | head -10

3. For each distinct error template:
   - `gh issue list --state open --search "<short error fingerprint>"` to find
     existing issues.
   - If an open issue covers it, add a comment with: current count, timestamp,
     one-line sample. Do not open a duplicate.
   - If no open issue exists, read the repo to locate the code that emits the
     line, then open a new issue with: title, sample, suspected file:line, and
     a one-paragraph hypothesis of the cause.

Do not edit code in this loop. Issues only.
```

## Launching the loop

From a Claude Code session in the repo:

```
/loop 30m /scan-logs
```

Every 30 minutes the agent pulls logs, dedupes them against open issues, and either files something new or comments on what's already tracked. Stop with `/loop stop`.

## Wrap-up

The whole thing is `logcli`, a read-only Loki token, `gh`, and a 20-line prompt. The agent never touches a VM and the only credentials it ever sees are the two read-only tokens you scoped for it.
