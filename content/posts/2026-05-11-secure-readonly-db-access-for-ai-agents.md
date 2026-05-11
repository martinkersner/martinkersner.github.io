---
title: "Secure read-only DB access for AI agents"
date: 2026-05-11T00:00:00Z
slug: "secure-readonly-db-access-for-ai-agents"
disqus_identifier: 2026-05-11
author: martin
summary: "AI agents are useful, but giving them unrestricted database access has real consequences. Cursor running Claude Opus 4.6 <a href='https://x.com/lifeof_jer/status/2048103471019434248' target='_blank'>recently wiped a production database in 9 seconds</a>. A bit of setup at the DB side prevents that. Below is a minimal recipe for PostgreSQL: a <code>SELECT</code>-only role with statement timeout and audit logging, <code>.pgpass</code> + <code>.pg_service.conf</code> so the password stays out of argv, and a Claude Code allowlist that pins the connection target. The same pattern maps to MySQL, MongoDB, BigQuery, and most other databases."
comments: true
---

Agentic engineering tools are useful, but unrestricted access has real consequences. Cursor running Claude Opus 4.6 <a href="https://x.com/lifeof_jer/status/2048103471019434248" target="_blank">recently wiped a production database in 9 seconds</a>. A bit of setup prevents that.

My use case is querying trading and market data (public numbers, no user records), so the blast radius I care about is destructive writes and runaway queries, not what ends up in the agent's transcript. If your schema holds PII or secrets, note that read-only ≠ private: whatever the agent reads can land in conversation logs. In that case, `GRANT SELECT` per-table rather than `ON ALL TABLES`.

Recipe below is for PostgreSQL. The same pattern (dedicated role, `SELECT`-only, statement timeout, narrow client allowlist) maps to MySQL, MongoDB, BigQuery, and most other databases.

## 1. Create a read-only role

```sql
CREATE ROLE claude WITH LOGIN PASSWORD 'mypassword';
GRANT CONNECT ON DATABASE mydb TO claude;
GRANT USAGE ON SCHEMA myschema TO claude;
GRANT SELECT ON ALL TABLES IN SCHEMA myschema TO claude;
REVOKE CREATE ON SCHEMA public FROM claude; -- PG ≤14 grants this by default via PUBLIC
ALTER ROLE claude SET statement_timeout = '15s'; -- starting value only; the session can SET it back to 0
ALTER ROLE claude SET default_transaction_read_only = true;
ALTER ROLE claude SET log_statement = 'all'; -- audit trail goes to the Postgres server log
ALTER ROLE claude SET temp_file_limit = '1GB'; -- cap disk spill from sort/hash queries
```

No `INSERT`, `UPDATE`, `DELETE`, `DROP`. Statement timeout kills runaway queries, though it's a `USERSET` parameter, so the agent can override it in-session; for hard enforcement, put a connection pooler in front or point the role at a read replica. `default_transaction_read_only` is a second layer: if a future `GRANT` accidentally adds write access, transactions still start in read-only mode.

## 2. Give Claude the connection details

Keep the password out of argv (and out of shell history, `ps`, and tool logs). Two files do the work. Replace `db.example.com` with the actual address (LAN IP, Tailscale IP, or hostname) of your Postgres server.

Put the password in `~/.pgpass` and `chmod 600` the file:

```
db.example.com:5432:mydb:claude:mypassword
```

Then add a named service entry to `~/.pg_service.conf` with the rest of the connection details:

```
[mydb]
host=db.example.com
port=5432
dbname=mydb
user=claude
sslmode=require
```

Now `psql service=mydb` connects with no flags and no secret on the command line.

If your Postgres is reachable over an unencrypted network, also force TLS at the server with a `hostssl` entry in `pg_hba.conf`. The client `sslmode=require` above only asks for TLS; `hostssl` makes the server refuse plaintext outright. Skip if you're already on Tailscale, WireGuard, or a VPN.

## 3. Allowlist the exact command

In `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(psql service=mydb)",
      "Bash(psql service=mydb *)"
    ]
  }
}
```

`service=mydb` pins host, port, db, user, and `sslmode` via the conf file, so the allowlist locks the whole connection target, not just the binary name.

## What about the MCP server?

Anthropic ships a reference Postgres MCP server that's read-only by design. The difference vs. this recipe is where the boundary lives: the MCP server enforces read-only in application code, while a role-based setup enforces it at the DB itself, the hardest possible boundary. The two compose well. Run the MCP server *with* the read-only role from this post and you get two locks instead of one: even if the server has a bug that exposes a write path, the DB still refuses.

## Wrap-up

I've been running this against my own Postgres for a few weeks. Claude has executed plenty of `SELECT`s through it, no surprises.
