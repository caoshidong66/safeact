# Payments Production Rollback Runbook

Before running rollback_execute in production, check the active deploy state, current incident lock, and schema compatibility. If schema compatibility reports a forward-only migration or incompatible ledger schema, rollback_execute is blocked. Escalate to the incident commander and DB owner instead.
