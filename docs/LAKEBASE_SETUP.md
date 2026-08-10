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

### 2.1 Job Task Run-As Identity Grants & Ownership Subtlety

The scheduled Databricks job notebook connects as its **run-as identity** (`WorkspaceClient().current_user.me().user_name`), which differs from the App's Service Principal:
- The job run-as identity requires `USAGE, CREATE ON SCHEMA capstone` and `CONNECT ON DATABASE databricks_postgres`.
- **Table Ownership Note**: `init_db()` executing inside the job task creates tables owned by the job run-as identity. Standard PostgreSQL `ALTER DEFAULT PRIVILEGES` only grants privileges on objects created by the role that executes that statement. Ensure default privileges are configured on the job run-as identity role as well so the Streamlit App retains access to newly created tables.

---

## 3. Local Development Environment Configuration

For local development against Lakebase using user OAuth credentials:

```bash
databricks auth login --host https://dbc-117d1e6a-753a.cloud.databricks.com

export PGHOST="ep-lingering-glitter-d87p4ci7.database.us-east-2.cloud.databricks.com"
export PGPORT="5432"
export PGDATABASE="databricks_postgres"
export PGUSER="your.email@company.com"
export PGSSLMODE="require"
export PGENDPOINT="projects/capstone-lakebase-new/branches/dev/endpoints/primary"
export PGSCHEMA="capstone"

uv run streamlit run app.py
```

---

## 4. GitHub Actions CI Secrets Configuration (`gh` CLI)

To configure workspace credentials for GitHub Actions CI bundle validation:

```bash
# 1. Set Databricks Workspace Host
gh secret set DATABRICKS_HOST --repo PedroLiu1999/dataexpert-bootcamp-capstone --body "https://dbc-117d1e6a-753a.cloud.databricks.com"

# 2. Set Databricks OAuth Access Token
gh secret set DATABRICKS_TOKEN --repo PedroLiu1999/dataexpert-bootcamp-capstone --body "$(databricks auth token | jq -r .access_token)"

# 3. List GitHub Repository Secrets
gh secret list --repo PedroLiu1999/dataexpert-bootcamp-capstone
```
