# Static public demo deployment

The working research demo is:
https://flora-rollout-review.exact-leaf-7339.chatgpt.site

It is a public static Sites deployment of this repository's `web/` files.
A clean Chrome Incognito window loaded the example, verified and played its
video, saved a separate correction, and downloaded its bundle without a
ChatGPT login. CLI validation accepted the downloaded history. The published robot example
also played with synchronized recorded actions and produced a real downloaded
bundle for CLI export and robomimic loader verification. The host applies
Cloudflare bot filtering; a Python URL request received HTTP403 while the
ordinary anonymous browser workflow passed. No bypass token is used by the app.

The page has no inference server, upload endpoint, external analytics, database,
or cluster connection. Public media files are explicitly licensed examples.
Own bundles are parsed in the browser. Selected own media are hashed and played
as Blob URLs. Corrections stay in browser storage until the user downloads a
bundle. Private/incognito storage is temporary; download before closing it.

## Source and updates

GitHub remains the source of record. `web/` is a buildless static app. Its tests
run in GitHub CI. The package also includes the same static resources. A Sites
mirror copies only `web/` to its static `dist/` folder, then uses the installed
Sites source helper to commit/package and deploy it. Site credentials and local
operational state are not part of this repository. Future updates must rerun
the frontend tests, preserve public access, and verify the new static files.

The repository also contains `.github/workflows/pages.yml`. Its first deployment
succeeded, but GitHub's project URL inherits the account's existing user-site
custom domain. It redirects to `iamayman.co`, which did not resolve during this
release. No account-level domain setting or unrelated repository was changed.
The alternate public host avoids that external dependency. No new domain,
paid API, or paid compute was purchased.
