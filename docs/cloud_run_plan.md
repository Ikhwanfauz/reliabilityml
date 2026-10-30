# ReliabilityML Cloud Run trial

Project: `reliabilityml-ikhwan-2026`
Region: `asia-southeast1`
Service: `reliabilityml-api` (private)
Image: `reliabilityml-api:deploy-v2` with the 48-feature model

Cloud Run used 1 CPU, 1 GiB memory, minimum 0 and maximum 1 instance,
concurrency 1, and port 8000.

## Verification

The image was pushed to Artifact Registry and deployed to Cloud Run.
Authenticated `/predict` for FD001 test engine 1 at cycle 31 returned
180.89 predicted RUL cycles. Unauthenticated `/health` returned HTTP 403.

The manual GitHub Actions CD workflow ran tests, authenticated using
GitHub OIDC, and deployed the existing image. Revision
`reliabilityml-api-00003-bl6` used `reliabilityml-runtime` and passed
the same authenticated prediction smoke test.

This workflow deploys an existing image; it does not build and push
a new image for each code change.

## Cleanup and current status

The Cloud Run service and Artifact Registry repository were deleted;
each list command returned 0 items. Project billing was unlinked and
verified as `billingEnabled: false`. The cloud endpoint is no longer live.
A future cloud deployment requires restoring billing, the repository,
and its image. Earlier usage charges may be reported later.

SQLite monitoring and drift checks were demonstrated locally. The cloud
trial did not include persistent monitoring storage or live failure labels.