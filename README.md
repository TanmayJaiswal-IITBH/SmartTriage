SmartTriage

An intelligent GitHub bot that automates issue triage and PR reviewer assignment using vector similarity search and commit history analysis for OpenLake repositories.
Status: In Development  |  License: MIT
Table of Contents

About the Project
Tech Stack
Architecture
Features
Project Structure
Getting Started
Quick Start Guide
Environment Variables Example
Basic API Overview
Common Issues & Troubleshooting
Contact
About the Project

↥ Back to top
SmartTriage is an AI-powered GitHub App designed to reduce maintainer overhead across OpenLake repositories. It automatically detects duplicate issues using vector similarity and intelligently routes pull requests to the most qualified reviewers based on commit history.
Problems Being Solved

Duplicate Issues: Automatically detects and closes duplicate issues by comparing them against historical data using semantic embeddings.
PR Reviewer Bottleneck: Recommends the right reviewers by analyzing which developers have most recently touched the files changed in a PR.
Maintainer Overhead: Reduces repetitive triage work so maintainers can focus on actual development.
Onboarding Friction: Helps new contributors navigate the codebase by surfacing related historical issues and the right points of contact.
Target Audience

OpenLake Maintainers — automating issue management and PR routing
Contributors — faster feedback and clearer reviewer assignments
Club Administrators — visibility into issue trends and project health
Tech Stack

↥ Back to top
Backend: Python, FastAPI, Uvicorn
ML / Embeddings: Hugging Face sentence-transformers
Vector Database: ChromaDB (local storage)
GitHub Integration: PyGithub, GitHub App Webhooks
Containerization: Docker
Version Control: Git and GitHub
Architecture

↥ Back to top
Backend

API Layer (app/api/): FastAPI webhook endpoint that receives GitHub events and verifies request signatures.
Core (app/core/): PyGithub wrapper for fetching issues, posting comments, and assigning reviewers.
ML (app/ml/): Sentence-transformer logic for generating text embeddings and mapping commit history to reviewer suggestions.
Database (app/db/): ChromaDB connection, vector insertion, and similarity search logic.
Config (app/config.py): Safe loading of environment variables and secrets.
Implementation Phases

The project is built in 5 phases to keep development manageable:
Authentication & Webhook Infrastructure — GitHub App registration, FastAPI setup, signature verification.
Historical Data Ingestion — Fetch, clean, embed, and store all historical OpenLake issues into ChromaDB.
Real-Time Duplicate Detection — Vectorize incoming issues, query ChromaDB for similar ones, auto-comment and close duplicates above a similarity threshold.
Reviewer Recommendation — Analyze changed files in a PR, map to git commit history, and auto-assign top contributors as reviewers.
Containerization & Deployment — Dockerize the app and deploy to a live server with the GitHub App webhook pointing to it.
Features

↥ Back to top
Duplicate Issue Detection: Converts new issue text to vector embeddings and queries historical issues; automatically comments and closes issues above an 85% similarity threshold.
Reviewer Recommendation: Analyzes files changed in a PR and recommends the top 2 developers with the most relevant commit history from the past 6 months.
Webhook Signature Verification: Ensures all incoming GitHub requests are legitimate before processing.
Historical Data Ingestion: One-time script to embed and store all past issues across major OpenLake repositories.
Automated Commenting: Posts context-aware comments on duplicate issues, linking back to the original.
Containerized Deployment: Docker setup for consistent local and production environments.
Project Structure

↥ Back to top
smart-triage/
├── .env                        # Secrets (GitHub App ID, Private Key, Webhook Secret)
├── .env.example                # Example env file for setup
├── .gitignore                  # Ignore /data, .env, __pycache__
├── requirements.txt            # FastAPI, PyGithub, chromadb, sentence-transformers
├── Dockerfile                  # Container configuration for easy deployment
├── README.md                   # Main project documentation
├── data/                       # Local ChromaDB storage (git ignored)
└── app/
    ├── __init__.py
    ├── main.py                 # FastAPI entry point (uvicorn runs this)
    ├── config.py               # Loads environment variables safely
    ├── api/
    │   └── webhooks.py         # Endpoint that GitHub hits when an issue is opened
    ├── core/
    │   └── github.py           # PyGithub wrapper (fetches data, posts comments)
    ├── ml/
    │   ├── embedder.py         # Hugging Face sentence-transformers logic
    │   └── reviewer.py         # Logic to map commit history to reviewer suggestions
    └── db/
        └── vector.py           # ChromaDB connection, insert, and search logic

Getting Started

↥ Back to top
Prerequisites

Python 3.10+
pip
Docker (optional, for containerized setup)
A registered GitHub App with Issues and Pull Requests read/write permissions
Installation

Clone the repository:
git clone https://github.com/OpenLake/smart-triagecd smart-triage

Install dependencies:
pip install -r requirements.txt

Set up environment variables:
cp .env.example .env# Edit .env with your GitHub App credentials

Run the historical data ingestion script (one-time setup):
python -m app.db.ingest

Start the FastAPI server:
uvicorn app.main:app --reload --port 8000

Expose your local server to GitHub using a tunneling tool (e.g., ngrok) for development:
ngrok http 8000

Update your GitHub App's webhook URL to point to:
https://<your-ngrok-url>/webhook

Quick Start Guide

↥ Back to top
For a quick local setup:
# Install dependencies
pip install -r requirements.txt# Copy and fill in environment variables
cp .env.example .env# Start the server
uvicorn app.main:app --reload --port 8000

Or with Docker:
docker build -t smart-triage .
docker run --env-file .env -p 8000:8000 smart-triage

Make sure your GitHub App webhook is pointed at your running server before testing.
Environment Variables Example

↥ Back to top
Create a .env file in the project root with the following format:
GITHUB_APP_ID=your_github_app_idGITHUB_PRIVATE_KEY=your_private_key_pem_contentsGITHUB_WEBHOOK_SECRET=your_webhook_secretSIMILARITY_THRESHOLD=0.85CHROMA_DB_PATH=./data

Refer to .env.example for all required variables.
Basic API Overview

↥ Back to top
EndpointMethodDescription/webhookPOSTReceives GitHub webhook events (issue opened, PR opened)/healthGETHealth check for the running server
Refer to app/api/webhooks.py for the full webhook routing logic.
Common Issues & Troubleshooting

↥ Back to top
Ensure your GitHub App has the correct Issues and Pull Requests read/write permissions.
The webhook secret in your .env must match exactly what is set in your GitHub App settings.
If ChromaDB returns no results, re-run the historical ingestion script — the data/ directory may be empty or missing.
When testing locally, make sure your tunneling tool (e.g., ngrok) is running and the webhook URL in your GitHub App settings is updated to the current tunnel URL.
Ensure SIMILARITY_THRESHOLD is set appropriately — values too low will trigger false positives; too high may miss actual duplicates.
Contact

↥ Back to top
If you have any questions or feedback, feel free to reach out to the maintainers or open an issue in the repository.

update this with current structure
