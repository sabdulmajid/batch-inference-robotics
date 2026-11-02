# Browser rollout review

This static app opens a versioned rollout bundle and separately selected source
videos. It keeps generated captions, review history, recorded actions, and
outcome evidence distinct. It does not run inference or train a policy.

Serve this directory with an HTTPS static host, or use localhost during development:

```sh
python3 -m http.server 8080 --directory web
```

Open `http://localhost:8080`. A direct `file://` open is not supported because
module loading and checksum verification need a secure browser context.
There are no build dependencies.

## Public example assets

`examples/index.json` lists permitted, precomputed examples:

```json
{
  "examples": [
    {
      "id": "example-id",
      "title": "A clear description of the source and available evidence",
      "bundle": "example.bir.json",
      "media": {"SHA256_OF_SOURCE_MEDIA": "example.mp4"},
      "attribution": "Dataset, media license, source link, and any use restrictions"
    }
  ]
}
```

All referenced assets must stay inside this site's `examples/` directory.
External URLs and redirects are rejected. The bundle and video checksums are
verified before display or playback. Examples must retain their real source
identities. Synthetic tests must never be presented as native model results.

## Own-file round trip

1. Select **Import bundle** and choose a `bir_rollout_bundle_v1` JSON file.
2. Select **Select local media** and explicitly choose the matching source videos.
   A filename alone does not establish identity; media bytes must match SHA-256.
3. Select a completed caption interval, inspect the video, enter a reviewer
   identifier, and save a correction. Pending and failed captions cannot receive
   new corrections. Earlier inapplicable edits remain visible in history.
4. Select **Download bundle**. The JSON file contains the unchanged prediction
   payload and complete append-only correction journal. Keep the original media
   beside it; media bytes are not embedded in the JSON file.
5. Reopen the downloaded bundle. Its correction history does not depend on this
   browser. Use the CLI to validate it and create a training export with an
   explicit reward source and boundary policy.

The app stores a recovery journal in local browser storage, keyed by the exact
payload checksum. It offers a verified later draft instead of silently replacing
an imported history. Competing histories are not merged. Browser storage can be
cleared or unavailable; the save status reports that state. A downloaded bundle
is the portable copy. A hash chain detects modification, but it is not a signed
attestation of a reviewer's identity.

## Data path and limits

User-selected bundles and media stay in the browser. The app contains no upload,
POST request, analytics, external script, or inference endpoint. It uses browser
File APIs and local blob URLs for user media. The only fetches load this site's
public example index, listed bundles, and listed videos. A content security
policy limits scripts, styles, and connections to this site and disallows forms,
objects, and embedded frames. Links to the repository open only on user action.

The app accepts bundles up to 64 MiB and each video up to 256 MiB. Bundle
validation checks canonical payload bytes, each generated-record identity,
trajectory identity, outcome request/result seals, review chain, all source
bindings, and supported clocks. A numeric-spelling scanner preserves Python
floating-point bytes while checking hashes; it never rewrites the payload. The CLI performs the authoritative export and
artifact validation; browser inspection does not establish model execution or
training suitability. Source media must use a codec supported by the browser.

Recorded trajectories use explicit video-to-trajectory clock mappings. The
selected transition is found from its recorded action time, not an equal array
index. Unknown episode ending types and missing Rewarder scores remain unknown.
A source environment reward is not a FLORA judgment. Test-fixture outcomes are
marked as fixtures even when their parser state is complete.

## Verification

```sh
node --test web/tests/*.test.mjs
```

These tests cover immutable payload bytes, correction/restore round trips,
hash-chain corruption, pending/failed captions, orphaned history, interval/source
mismatches, clock consistency, outcome source binding, stale-identity attacks,
and fixture outcome labels. A Python-generated Unicode/float fixture checks
cross-language identity handling. Browser checks must
also verify playback, interval selection, local-file privacy, actual download,
reopen, keyboard focus, and responsive layout against the deployed app.
