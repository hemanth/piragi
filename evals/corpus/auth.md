# Authentication Service

The `auth-gateway` module issues JWT access tokens with a 15 minute TTL and
refresh tokens with a 30 day TTL. Invalid API keys return error code
`AUTH_4011`. The rate limiter allows 100 requests per minute per API key
before returning `AUTH_4291` (rate limited).

To rotate a leaked API key, call `POST /v1/keys/rotate` with the old key in
the `X-Old-Key` header. Rotation invalidates the old key within 60 seconds.
