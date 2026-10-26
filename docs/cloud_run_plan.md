# ReliabilityML Cloud Run trial plan

Project: reliabilityml-ikhwan-2026
Region: asia-southeast1 (Singapore)
Service: reliabilityml-api
Artifact Registry repository: reliabilityml
Image: reliabilityml-api:deploy-v2 (includes the 48-feature model)

Proposed Cloud Run settings:
- CPU: 1
- Memory: 1 GiB
- Minimum instances: 0
- Maximum instances: 1
- Concurrency: 1
- Container port: 8000
- Access: private for the first deployment

Local evidence:
- /health returned {"status":"ok"}.
- /predict returned 180.89 cycles for test engine 1 at cycle 31.
- The prediction succeeded with a 1 GiB Docker memory limit.
- Memory measured after that prediction: 259.8 MiB.

Cost and cleanup:
- Check current GCP pricing and billing status before deployment.
- Artifact Registry stores the uploaded image until its repository is deleted.
- After the cloud smoke test, delete the Cloud Run service and Artifact Registry repository.
- Verify both are gone, then unlink billing from this project if it was linked.
- A budget alert alone does not stop charges.

Limitations:
- The local SQLite monitoring database is not persistent Cloud Run storage.
- This trial tests cloud inference, not persistent cloud monitoring.
- The local RAM measurement is not a peak-load guarantee.

## Windows Anaconda Prompt setup

If `where gcloud` cannot find Google Cloud CLI, run this in the current Prompt:

    set "PATH=%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin;%PATH%"

For a private Cloud Run smoke test, set `RELIABILITYML_API_URL` to the
service URL ending in `/predict` and set `RELIABILITYML_AUTH=gcloud`.
The smoke script obtains an ID token without printing or saving it.
Clear the auth setting afterwards with `set RELIABILITYML_AUTH=`.