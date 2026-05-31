# Google Search Console API Notes

Use this file for endpoint, auth, quota, and query-shape reminders.

## API surface

The official Search Console API covers:

- Search Analytics: traffic data grouped by dimensions such as query, page, country, device, date, and hour.
- Sitemaps: list, get, submit, and delete sitemaps.
- Sites: list, get, add, and delete Search Console properties.
- URL Inspection: inspect Google's indexed version of a URL.

The Search Console web UI is not required for reads once OAuth is configured, but the Google account must have permission on the Search Console property.

## Base URLs

Search Analytics, Sites, and Sitemaps use:

```text
https://www.googleapis.com/webmasters/v3
```

URL Inspection uses:

```text
https://searchconsole.googleapis.com/v1
```

Property strings are path parameters and must be URL-encoded in request URLs. Domain properties look like `sc-domain:example.com`. URL-prefix properties look like `https://www.example.com/` and often require the trailing slash.

## OAuth

Search Console API requests require OAuth 2.0.

Scopes:

- `https://www.googleapis.com/auth/webmasters.readonly` for reads.
- `https://www.googleapis.com/auth/webmasters` for read/write access.

For headless machines, use the browser URL flow. Create an OAuth **Desktop app** client in Google Cloud, enable the Search Console API for the project, then run:

```bash
<skill-path>/scripts/gsc-cli auth url --client-id '<client-id>'
```

The command prints an auth URL. Open it in a normal browser. The redirect to `http://localhost` may fail on your laptop; copy the final URL from the browser address bar and paste it back into the CLI.

Google's OAuth device flow may reject Search Console scopes with `invalid_scope`, so do not rely on TV/Limited Input clients for this API.

Common setup failures:

- `invalid_scope` during `auth device`: use `auth url` with a Desktop OAuth client instead.
- `SERVICE_DISABLED` / `accessNotConfigured`: enable the Search Console API for the numeric project shown in the error's activation URL, then retry after propagation.
- Refresh token expires after about a week: the OAuth app is probably External + Testing. Use an Internal app for the Workspace org, or move the app to production if appropriate.
- `sites list` is empty: the authenticated Google account has no Search Console permissions for those properties, or the wrong account was selected during OAuth.
- `403` on a specific property: check exact property string (`sc-domain:example.com` versus URL-prefix `https://www.example.com/`) and permission level.

## Search Analytics

Endpoint:

```text
POST /sites/{siteUrl}/searchAnalytics/query
```

Required request fields:

- `startDate`: `YYYY-MM-DD`
- `endDate`: `YYYY-MM-DD`

Useful optional fields:

- `dimensions`: ordered list such as `query,page`, `page,date`, `country,device`.
- `type`: `web`, `image`, `video`, `news`, `discover`, or `googleNews`.
- `dimensionFilterGroups`: filters over `country`, `device`, `page`, `query`, or `searchAppearance`.
- `aggregationType`: `auto`, `byPage`, or `byProperty`.
- `rowLimit`: 1 to 25000.
- `startRow`: zero-based offset.

Filter operators:

- `contains`
- `equals`
- `notContains`
- `notEquals`
- `includingRegex`
- `excludingRegex`

Search Analytics returns top rows, not guaranteed exhaustive full-table exports. Expensive queries group or filter by `page` and `query`, especially across long ranges.

## URL Inspection

Endpoint:

```text
POST https://searchconsole.googleapis.com/v1/urlInspection/index:inspect
```

Request body:

```json
{
  "inspectionUrl": "https://www.example.com/page",
  "siteUrl": "https://www.example.com/",
  "languageCode": "en-US"
}
```

The API reports the URL status in Google's index. It does not test the live page's current indexability.

There is no Search Console API endpoint for the UI's manual "Request indexing" action for arbitrary normal pages. The full Search Console scope supports sitemap/site mutations. For normal new or updated pages, submit or update sitemaps and make sure `<lastmod>` is accurate.

## Sitemaps

Endpoints:

```text
GET    /sites/{siteUrl}/sitemaps
GET    /sites/{siteUrl}/sitemaps/{feedpath}
PUT    /sites/{siteUrl}/sitemaps/{feedpath}
DELETE /sites/{siteUrl}/sitemaps/{feedpath}
```

`feedpath` is the sitemap URL and must be URL-encoded.

## Sites

Endpoints:

```text
GET    /sites
GET    /sites/{siteUrl}
PUT    /sites/{siteUrl}
DELETE /sites/{siteUrl}
```

Site rows include `siteUrl` and `permissionLevel`, such as `siteOwner`, `siteFullUser`, `siteRestrictedUser`, or `siteUnverifiedUser`.

## Quotas

Search Analytics has load quotas and request quotas. Broad date ranges and grouping/filtering by both `page` and `query` are more expensive.

URL Inspection has per-site quota. Avoid bulk inspection loops unless the user explicitly asks and the quota impact is acceptable.

## Official docs

- API reference: https://developers.google.com/webmaster-tools/v1/api_reference_index
- Prerequisites: https://developers.google.com/webmaster-tools/v1/prereqs
- Authorization: https://developers.google.com/webmaster-tools/v1/how-tos/authorizing
- Search Analytics query: https://developers.google.com/webmaster-tools/v1/searchanalytics/query
- Sitemaps: https://developers.google.com/webmaster-tools/v1/sitemaps
- Sites: https://developers.google.com/webmaster-tools/v1/sites
- URL Inspection index.inspect: https://developers.google.com/webmaster-tools/v1/urlInspection.index/inspect
- Usage limits: https://developers.google.com/webmaster-tools/limits
