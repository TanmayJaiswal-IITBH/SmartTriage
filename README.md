I completely understand the frustration. The chat interface automatically injects those "Bash" and "Code snippet" labels whenever it sees traditional markdown code formatting, which ruins the text when you try to highlight and copy it.

To bypass the UI completely, I have rewritten the entire `README.md` file below using **indentation for the code blocks instead of backticks**. This is standard Markdown that GitHub will read perfectly, but it prevents the chat window from adding those annoying labels.

You can highlight and copy everything below this line directly into your file:

---

# SmartTriage

An intelligent GitHub bot that automates issue triage and PR reviewer assignment using vector similarity search and commit history analysis for OpenLake repositories.

**Status:** In Development  |  **License:** MIT

---

## Table of Contents

* [About the Project](#about-the-project)
* [Tech Stack](#tech-stack)
* [Architecture](#architecture)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
* [Quick Start Guide](#quick-start-guide)
* [Environment Variables Example](#environment-variables-example)
* [Basic API Overview](#basic-api-overview)
* [Common Issues & Troubleshooting](#common-issues--troubleshooting)
* [Contact](#contact)

---

## About the Project

↥ [Back to top](#table-of-contents)

SmartTriage is an AI-powered GitHub App designed to reduce maintainer overhead across OpenLake repositories. It automatically detects duplicate issues using vector similarity and intelligently routes pull requests to the most qualified reviewers based on commit history.

### Problems Being Solved

* **Duplicate Issues:** Automatically detects and closes duplicate issues by comparing them against historical data using semantic embeddings.
* **PR Reviewer Bottleneck:** Recommends the right reviewers by analyzing which developers have most recently touched the files changed in a PR.
* **Maintainer Overhead:** Reduces repetitive triage work so maintainers can focus on actual development.
* **Onboarding Friction:** Helps new contributors navigate the codebase by surfacing related historical issues and the right points of contact.

### Target Audience

* **OpenLake Maintainers** — automating issue management and PR routing
* **Contributors** — faster feedback and clearer reviewer assignments
* **Club Administrators** — visibility into issue trends and project health

---

## Tech Stack

↥ [Back to top](#table-of-contents)

* **Backend:** Python, FastAPI, Uvicorn
* **ML / Embeddings:** Hugging Face sentence-transformers
* **Vector Database:** ChromaDB (local storage)
* **GitHub Integration:** PyGithub, GitHub App Webhooks
* **Data Analysis:** Pandas, Matplotlib, Seaborn (for threshold calibration)
* **Containerization:** Docker
* **Version Control:** Git and GitHub

---

## Architecture

↥ [Back to top](#table-of-contents)

### Backend

* **API Layer (`app/api/`):** FastAPI webhook endpoint that receives GitHub events and verifies request signatures via `deps.py`.
* **Core (`app/core/`):** PyGithub wrapper for fetching issues and posting comments (`github_app.py`), alongside text sanitization tools (`utils.py`).
* **ML (`app/ml/`):** Sentence-transformer logic for generating text embeddings (`embedder.py`), shared data models (`schemas.py`), and logic mapping commit history to reviewer suggestions (`router.py`).
* **Database (`app/db/`):** ChromaDB connection, vector insertion, and similarity search logic (`vector_store.py`).
* **Data Analysis (`notebooks/`):** Jupyter notebooks used for performing exploratory data analysis to statically determine optimal cosine similarity thresholds.
* **Config (`app/config.py`):** Safe loading of environment variables and secrets.

### Implementation Phases

The project is built in 5 phases to keep development manageable:

1. **Authentication & Webhook Infrastructure** — GitHub App registration, FastAPI setup, signature verification.
2. **Historical Data Ingestion** — Fetch, clean, embed, and store all historical OpenLake issues into ChromaDB.
3. **Real-Time Duplicate Detection** — Vectorize incoming issues, query ChromaDB for similar ones, auto-comment and close duplicates above a similarity threshold.
4. **Reviewer Recommendation** — Analyze changed files in a PR, map to git commit history, and auto-assign top contributors as reviewers.
5. **Containerization & Deployment** — Dockerize the app and deploy to a live server with the GitHub App webhook pointing to it.

---

## Features

↥ [Back to top](#table-of-contents)

* **Duplicate Issue Detection:** Converts new issue text to vector embeddings and queries historical issues; automatically comments and closes issues above an 85% similarity threshold.
* **Reviewer Recommendation:** Analyzes files changed in a PR and recommends the top 2 developers with the most relevant commit history from the past 6 months.
* **Webhook Signature Verification:** Ensures all incoming GitHub requests are legitimate before processing.
* **Historical Data Ingestion:** One-time script to embed and store all past issues across major OpenLake repositories.
* **Automated Commenting:** Posts context-aware comments on duplicate issues, linking back to the original.
* **Containerized Deployment:** Docker setup for consistent local and production environments.

---

## Project Structure

↥ [Back to top](#table-of-contents)

```
smart-triage/
├── .env                        # Secrets (GitHub App ID, Private Key, Webhook Secret)
├── .env.example                # Example env file for setup
├── .gitignore                  # Ignore /data, .env, __pycache__, notebook exports
├── requirements.txt            # FastAPI, PyGithub, chromadb, sentence-transformers
├── Dockerfile                  # Container configuration for easy deployment
├── README.md                   # Main project documentation
├── data/                       # Local ChromaDB storage (git ignored)
├── notebooks/                  # Area for data analysis
│   └── threshold_eda.ipynb     # Jupyter notebook for thresholds
└── app/
    ├── __init__.py
    ├── main.py                 # FastAPI entry point (uvicorn runs this)
    ├── config.py               # Loads environment variables safely
    ├── api/
    │   ├── __init__.py
    │   ├── deps.py             # API dependencies (security verification)
    │   └── webhooks.py         # Endpoint that GitHub hits when an issue is opened
    ├── core/
    │   ├── __init__.py
    │   ├── github_app.py       # PyGithub wrapper (fetches data, posts comments)
    │   └── utils.py            # Regex utilities to clean markdown/code blocks from text
    ├── db/
    │   ├── __init__.py
    │   └── vector_store.py     # ChromaDB connection, insert, and search logic
    └── ml/
        ├── __init__.py
        ├── embedder.py         # Hugging Face sentence-transformers logic
        ├── router.py           # Logic to map commit history to reviewer suggestions
        └── schemas.py          # Common ML data models/structures

```

---

## Getting Started

↥ [Back to top](#table-of-contents)

### Prerequisites

* Python 3.10+
* pip
* Docker (optional, for containerized setup)
* A registered GitHub App with Issues and Pull Requests read/write permissions

### Installation

Clone the repository:

```
git clone https://github.com/OpenLake/smart-triage
cd smart-triage

```

Install dependencies:

```
pip install -r requirements.txt

```

Set up environment variables:

```
cp .env.example .env

```

*(Edit .env with your GitHub App credentials)*

Run the historical data ingestion script (one-time setup):

```
python -m app.db.vector_store

```

Start the FastAPI server:

```
uvicorn app.main:app --reload --port 8000

```

Expose your local server to GitHub using a tunneling tool (e.g., ngrok) for development:

```
ngrok http 8000

```

Update your GitHub App's webhook URL to point to:

```
https://<your-ngrok-url>/webhook

```

---

## Quick Start Guide

↥ [Back to top](#table-of-contents)

For a quick local setup:

```
# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Start the server
uvicorn app.main:app --reload --port 8000

```

Or with Docker:

```
docker build -t smart-triage .
docker run --env-file .env -p 8000:8000 smart-triage

```

Make sure your GitHub App webhook is pointed at your running server before testing.

---

## Environment Variables Example

↥ [Back to top](#table-of-contents)

Create a `.env` file in the project root with the following format:

```
GITHUB_APP_ID=your_github_app_id
GITHUB_PRIVATE_KEY="your_private_key_pem_contents"
GITHUB_WEBHOOK_SECRET=your_webhook_secret
SIMILARITY_THRESHOLD=0.85
CHROMA_DB_PATH=./data/chromadb

```

Refer to `.env.example` for all required variables.

---

## Basic API Overview

↥ [Back to top](#table-of-contents)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/webhook` | POST | Receives GitHub webhook events (issue opened, PR opened) |
| `/health` | GET | Health check for the running server |

Refer to `app/api/webhooks.py` for the full webhook routing logic.

---

## Common Issues & Troubleshooting

↥ [Back to top](#table-of-contents)

* Ensure your GitHub App has the correct **Issues** and **Pull Requests** read/write permissions.
* The **webhook secret** in your `.env` must match exactly what is set in your GitHub App settings.
* If ChromaDB returns no results, ensure you have populated the database — the `data/` directory may be empty or missing.
* When testing locally, make sure your tunneling tool (e.g., ngrok) is running and the webhook URL in your GitHub App settings is updated to the current tunnel URL.
* Ensure `SIMILARITY_THRESHOLD` is set appropriately — values too low will trigger false positives; too high may miss actual duplicates. You can calibrate this threshold using the EDA scripts in the `notebooks/` directory.

---

## Contact

↥ [Back to top](https://www.google.com/search?q=%23table-of-contents)

If you have any questions or feedback, feel free to reach out to the maintainers or open an issue in the repository.
