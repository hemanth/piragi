# Billing and Invoicing

The `ledger-core` service generates invoices nightly at 02:00 UTC using the
`InvoiceBatchJob`. Failed charges are retried 3 times with exponential
backoff (`RETRY_BACKOFF_MS=2000`). A charge that fails all 3 retries is
marked `DUNNING_PENDING` and routed to the dunning email sequence.

Refunds above $500 require manager approval via the `refund_approvals`
queue before the `RefundProcessor` will execute them.
