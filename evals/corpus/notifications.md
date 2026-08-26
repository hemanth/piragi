# Notification Service

The `notify-hub` fans out events to email, SMS, and push channels through
the `ChannelRouter`. SMS delivery uses the `TwilioAdapter` and retries up to
2 times on a `PROVIDER_TIMEOUT`. Push notifications older than 5 minutes are
dropped rather than delivered late, controlled by `PUSH_TTL_SECONDS=300`.

Users can mute a notification category by setting
`preferences.muted_categories` via `PATCH /v1/users/{id}/preferences`.
