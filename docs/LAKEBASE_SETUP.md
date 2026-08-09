# Lakebase PostgreSQL Setup & Grant Runbook

This document covers operational setup, identity grants, and local development configuration for Lakebase PostgreSQL in Databricks Apps.

---

## 1. Initial Bundle Deployment

Deploy the Databricks Asset Bundle first so the App, its Service Principal, and the Lakebase database branch exist before objects are created:

```bash
databricks bundle deploy --target dev
```

---

## 2. SQL Identity Grants (Lakebase SQL Editor)

In Databricks Lakebase SQL Editor, run the following to enable both the App's Service Principal (`<DATABRICKS_CLIENT_ID>`) and local developer identity (`<YOUR_EMAIL>`):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS capstone;

-- Grant schema access to both App Service Principal and developer identity
GRANT USAGE, CREATE ON SCHEMA capstone TO "<DATABRICKS_CLIENT_ID>";
GRANT USAGE, CREATE ON SCHEMA capstone TO "your.email@company.com";

-- Set default privileges so future tables created by either identity are accessible to the app
ALTER DEFAULT PRIVILEGES IN SCHEMA capstone
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "<DATABRICKS_CLIENT_ID>";
ALTER DEFAULT PRIVILEGES IN SCHEMA capstone
  GRANT USAGE, SELECT ON SEQUENCES TO "<DATABRICKS_CLIENT_ID>";

-- If connecting without attached resource (e.g. external driver / Spark identity)
CREATE EXTENSION IF NOT EXISTS databricks_auth;
SELECT databricks_create_role('<DATABRICKS_CLIENT_ID>', 'service_principal');
GRANT CONNECT ON DATABASE databricks_postgres TO "<DATABRICKS_CLIENT_ID>";
GRANT USAGE ON SCHEMA capstone TO "<DATABRICKS_CLIENT_ID>";
```

---

## 3. Local Development Environment Configuration

For local development against Lakebase using user OAuth credentials:

```bash
databricks auth login --host https://<workspace>.cloud.databricks.com

export PGHOST="ep-capstone-primary.database.cloud.databricks.com"
export PGPORT="5432"
export PGDATABASE="databricks_postgres"
export PGUSER="your.email@company.com"
export PGSSLMODE="require"
export PGENDPOINT="projects/capstone-lakebase/branches/dev/endpoints/primary"
export PGSCHEMA="capstone"

uv run streamlit run app.py
```
