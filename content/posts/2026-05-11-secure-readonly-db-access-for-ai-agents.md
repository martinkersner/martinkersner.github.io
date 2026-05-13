---
title: "Secure read-only DB access for AI agents"
date: 2026-05-11T00:00:00Z
slug: "secure-readonly-db-access-for-ai-agents"
disqus_identifier: 2026-05-11
author: martin
summary: "Cursor running Claude Opus 4.6 <a href='https://x.com/lifeof_jer/status/2048103471019434248' target='_blank'>wiped a production database</a> in 9 seconds. I keep Claude on a read-only Postgres role, with the password in a service file and an allowlist that pins the connection target. The risk I care about, for trading and market data (public numbers, no user records), is destructive writes and runaway queries."
comments: true
---

Cursor running Claude Opus 4.6 <a href="https://x.com/lifeof_jer/status/2048103471019434248" target="_blank">wiped a production database</a> in 9 seconds. I keep Claude on a read-only Postgres role, with the password in a service file and an allowlist that pins the connection target. The risk I care about, for trading and market data (public numbers, no user records), is destructive writes and runaway queries.

## 1. Create a read-only role

```sql
CREATE ROLE claude WITH LOGIN PASSWORD 'mypassword';
GRANT CONNECT ON DATABASE mydb TO claude;
GRANT USAGE ON SCHEMA myschema TO claude;
GRANT SELECT ON ALL TABLES IN SCHEMA myschema TO claude;
ALTER DEFAULT PRIVILEGES FOR ROLE owner IN SCHEMA myschema
  GRANT SELECT ON TABLES TO claude; -- cover tables created later
REVOKE CREATE ON SCHEMA public FROM claude; -- PG ≤14 grants this by default via PUBLIC
ALTER ROLE claude SET statement_timeout = '15s'; -- starting value only; the session can SET it back to 0
ALTER ROLE claude SET default_transaction_read_only = true;
ALTER ROLE claude SET log_statement = 'all'; -- audit trail goes to the Postgres server log
ALTER ROLE claude SET temp_file_limit = '1GB'; -- cap disk spill from sort/hash queries
```

No `INSERT`, `UPDATE`, `DELETE`, `DROP`. Statement timeout kills runaway queries, though it's a `USERSET` parameter, so the agent can override it in-session; for hard enforcement, put a connection pooler in front or point the role at a read replica.

`GRANT SELECT ON ALL TABLES` only covers tables that exist at grant time. `ALTER DEFAULT PRIVILEGES` is the standing rule that applies to anything `owner` creates later, so you don't have to re-grant on every new table. Replace `owner` with whichever role actually creates tables in the schema. If multiple roles create tables, repeat the `ALTER DEFAULT PRIVILEGES` line for each.

## 2. Give Claude the connection details

Keep the password out of argv (and out of shell history, `ps`, and tool logs). Two files do the work. Replace `db.example.com` with the actual address of your Postgres server.

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

Without `service=mydb`, the allowlist would let Claude run `psql` against anything.
