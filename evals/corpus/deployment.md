# Deployment Pipeline

The `deploy-orchestrator` runs canary deployments to 5% of pods for 10
minutes before promoting to 100%. A canary is rolled back automatically if
the `p99_latency_ms` metric exceeds 800ms or the error rate exceeds 1%.

Manual rollback is triggered with `deployctl rollback --service <name>
--to <revision>`. Rollbacks complete in under 90 seconds.
