# Zero-budget public deployment

## What this deployment does

The local application remains the default. `streamlit run app.py` continues to use the
existing offline NLLB/Canopy integrations and the Windows SAPI voice where available.

The separate `cloud/app.py` entry point starts the existing Streamlit application in
Cloudflare mode automatically.
This mode uses the Cloudflare Workers AI REST API with `@cf/meta/m2m100-1.2b` for text
translation. It does not import or load NLLB or Canopy, and it does not download model
weights at startup. The existing diacritic tools and wordbook remain local and usable.
The deployment requirements are in `cloud/requirements.txt`; the local `requirements.txt`
is not replaced.

The provider page documents M2M100 and examples for the REST interface. Meta's M2M100
model card lists Sindhi (`sd`) among its 100 languages. This establishes language
coverage, not translation quality or guaranteed Perso-Arabic spelling. The app checks
for Arabic-derived characters in Sindhi output, but that is only a script check. No
calibrated confidence score or accurate-translation claim is made. The endpoint returns
one result; the app does not manufacture alternatives. Cloudflare links the M2M100 model
to the fairseq MIT license; this deployment does not use or redistribute the local
NLLB checkpoint or its separate CC-BY-NC license.

## Speech and audio limitations

There is no hosted Sindhi ASR backend enabled. Cloudflare offers a Whisper ASR endpoint,
but the official model material found for this audit does not establish reliable
Sindhi-specific transcription or Perso-Arabic output. The public deployment therefore
does not record, retain, or transmit audio. `src/speech/providers.py` is the extension
interface for a future candidate; a provider must be reviewed and tested before it is
registered as verified. OpenAI's Whisper code is MIT-licensed, but that license and a
generic language parameter do not prove Sindhi ASR accuracy.

Canopy-M remains available only through the local app and its existing memory guard.
The existing English SAPI playback is Windows-local and is not available in the
Linux-hosted public app. No hosted TTS service is configured.

## Free-service terms and limits

- Cloudflare's Workers AI pricing page documents **10,000 Neurons per day** on the
  Workers Free plan. The daily allocation resets at 00:00 UTC; over-limit calls fail
  on the Free plan. The Workers Paid plan charges above the allocation. This app has
  no paid fallback and does not change account plans.
- The Cloudflare limits page lists 720 translation requests per minute. This is a
  provider-side upper limit, not a usage allowance or a promise of availability.
- The M2M100 page lists unit pricing, and the general Workers AI page documents the
  free allocation. Free usage is limited, shared with any other Workers AI use on the
  account, and may be changed by Cloudflare. Public visitors can exhaust the allowance;
  translation will then stop until the reset.
- The documentation identifies some models that require paid billing; M2M100 is not
  among that list. Cloudflare documents a default Free plan, but its retrieved signup
  documentation does not make an explicit guarantee that account registration will
  never request payment details. If Cloudflare asks for a card or requires billing
  activation, stop and do not continue. Do not attach a payment method, enable a Paid
  plan, or configure AI Gateway prepaid credits.
- Streamlit Community Cloud provides free hosting, but resource availability is shared
  and subject to change. Its current documentation describes approximately 0.69–2.7 GB
  RAM and 0.078–2 CPU cores (limits listed as of February 2024), and says apps without
  traffic for 12 hours sleep. It runs Debian Linux, requires GitHub access to the
  repository, and users can wake a sleeping public app. These are not an uptime or
  performance guarantee.
- The Streamlit Cloud host is in the United States. Text users choose to translate is
  sent from the app server to Cloudflare; do not submit confidential or sensitive text.
  The app stores no submitted translations or feedback in cloud mode.

## Credentials

Create a Cloudflare Workers AI API token with only the permissions required by the
Workers AI REST API, and keep both values in the Streamlit Community Cloud secret store:

```toml
CLOUDFLARE_ACCOUNT_ID = "your-account-id"
CLOUDFLARE_API_TOKEN = "your-workers-ai-token"
CLOUDFLARE_FREE_PLAN_CONFIRMED = "true"
```

Never commit these values or a local `.streamlit/secrets.toml`. The `.gitignore` excludes
the local secrets file. The API token is used only by the Streamlit server and is not
returned to the browser. Missing credentials produce an explicit error. Authentication,
network, model, and quota failures do not trigger a paid provider or local-model fallback.
The app refuses to send requests until the operator explicitly confirms the Cloudflare
account is on the Free plan with billing disabled. This confirmation is an operator
attestation, not an API-verified billing-status check; if the account is upgraded or
payment is enabled later, remove the confirmation and disable the remote mode.

## Publish and deploy

1. Review the project changes and confirm the public GitHub repository contains no
   credentials, private feedback, or files you do not want public. NLLB weights remain
   excluded by `.gitignore`; the small 25 KB diacritic classifier is intended to be
   included so the deployed restorer can reuse it. Training/test CSV files already in
   the repository remain part of the public source.
2. Do not use `git add .` in this worktree: there are pre-existing, uncommitted files
   and reports from earlier work. Stage only the reviewed code, tests, docs, cloud
   deployment files, and the small classifier. Inspect the staged diff before committing.
3. Push the reviewed commit to a GitHub branch you control. No push or public deployment
   has been performed by this change.
4. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect the
   GitHub account that can access the repository.
5. Select **Create app**, choose the repository and branch, and set the app file path to
   `cloud/app.py`. The `cloud/requirements.txt` file is next to that entry point, so the
   cloud build does not install PyTorch, Transformers, or Canopy.
6. In **Advanced settings → Secrets**, add the TOML values above. Choose Python 3.10 or
   another supported version compatible with the project's scikit-learn artifact.
   Do not add a Cloudflare payment method or select a paid plan. If account setup demands
   a card, stop rather than continuing.
7. Deploy, review the Cloud logs, and test with non-sensitive sample text. Only after a
   successful deployment will Streamlit provide the public `*.streamlit.app` URL. The
   deployment cannot be called live until that step and a browser smoke test are complete.

## Local checks

Run the app on Windows with the existing environment:

```powershell
streamlit run app.py
```

For local smoke testing of cloud-mode UI without calling the provider, set the three
secrets in a temporary local `.streamlit/secrets.toml` (never commit it), then run:

```powershell
streamlit run cloud/app.py
```

The automated provider tests mock HTTP calls and do not transmit text or audio:

```powershell
python -m pytest tests/test_cloudflare_provider.py -q
```

An authenticated, live translation test and a native-reviewed Sindhi quality/script
evaluation remain unverified until the account owner configures the provider and
approves a non-sensitive test phrase.

## Official sources

- [Cloudflare Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Cloudflare Workers AI limits](https://developers.cloudflare.com/workers-ai/platform/limits/)
- [Cloudflare M2M100 model and REST parameters](https://developers.cloudflare.com/workers-ai/models/m2m100-1.2b/)
- [Meta M2M100 model card and language list](https://huggingface.co/facebook/m2m100_1.2B)
- [Cloudflare Whisper ASR model](https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/)
- [OpenAI Whisper model card and limitations](https://github.com/openai/whisper/blob/main/model-card.md)
- [Streamlit Community Cloud resource limits and sleep behavior](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
