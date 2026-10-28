# English-to-Sindhi Neural Translator & Diacritic Restorer

AI-powered English-to-Sindhi translation and Sindhi diacritic restoration using Python, Streamlit, NLLB-200, and Cloudflare Workers AI.

## Live Demo

[Open the Translator](https://english-to-sindhi-translator-jrgxmrvx7sbjzwxfzryln7.streamlit.app/)

## Overview

This project provides a language translation application designed to help users translate English text into Sindhi and improve Sindhi text readability through diacritic restoration.

It includes a local application and a lightweight cloud application for Streamlit deployment.

## Features

- English-to-Sindhi neural translation.
- Sindhi diacritic restoration.
- Right-to-left Sindhi text display.
- Language selection supported by the configured provider.
- Copy translation controls.
- Responsive Streamlit user interface.
- Local translation using NLLB-200 Distilled 600M.
- Cloud translation using Cloudflare Workers AI.
- Automated tests using Pytest.

## Technology Stack

- Python
- Streamlit
- Hugging Face Transformers
- NLLB-200
- Cloudflare Workers AI
- Joblib
- Pytest
- Git and GitHub

## Project Structure

- `app.py` — Local application.
- `cloud/app.py` — Cloud application entry point.
- `cloud/requirements.txt` — Cloud dependencies.
- `src/translation/` — Translation provider code.
- `src/speech/` — Speech-related components.
- `models/diacritics/` — Diacritic restoration component.
- `tests/` — Automated tests.
- `docs/` — Project documentation.

## Run Locally

Python 3.10 is used for the existing local setup.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in your terminal, usually `http://localhost:8501`.

The local neural translation model requires sufficient RAM and disk space.

## Deploy to Streamlit Community Cloud

1. Connect this GitHub repository to Streamlit Community Cloud.
2. Select the `main` branch.
3. Set the main file path to `cloud/app.py`.
4. Configure the required Cloudflare credentials through Streamlit Secrets.
5. Select a Python version compatible with `cloud/requirements.txt`.

The cloud application avoids loading the large local translation model.

## Configuration and Security

Never commit API tokens, passwords, or private credentials. Configure Cloudflare credentials through deployment secrets.

Cloud translation requires provider access and is subject to service availability and usage limits. Verify the current free-plan terms before deployment.

## Limitations

- Translation results may require human review.
- Diacritic restoration is not guaranteed to be perfect.
- Local model performance depends on available hardware.
- Cloud translation requires valid configuration and internet access.
- Local speech functionality is not automatically available in the hosted version.

## Testing

Run the tests with:

```powershell
python -m pytest -q
```

Check Python syntax with:

```powershell
python -m py_compile app.py cloud/app.py
```

## Author

GitHub: [Akash-59](https://github.com/Akash-59)

## License

No license is specified here. Check the repository for an existing license before reusing or redistributing the project.
