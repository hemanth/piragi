# Caching Layer

The `redis-cache` layer stores session data with a default TTL of 3600
seconds, configurable via `CACHE_TTL_SECONDS`. Cache keys are namespaced as
`sess:{tenant_id}:{user_id}`. On a cache miss, the `CacheWarmupWorker`
repopulates the key from Postgres within 200ms.

Setting `CACHE_TTL_SECONDS=0` disables expiry entirely, which is only
recommended for the `analytics-readonly` tenant.
