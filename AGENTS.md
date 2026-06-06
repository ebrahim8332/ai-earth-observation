# AGENTS.md â€” AI Earth Observation Lab

This file applies to all work in this project when using OpenAI Codex or any other AI coding agent.

---

## Project Purpose

Multi-app platform for AI-native geospatial analysis. Each folder in `apps/` is a separate Streamlit application. Apps are built in daily sprints. Each sprint produces one working app or notebook.

---

## Repo Structure

```
ai-earth-observation/
â”œâ”€â”€ apps/           One subfolder per app. Self-contained.
â”œâ”€â”€ notebooks/      Exploratory analysis and data tests.
â”œâ”€â”€ docs/           Build specs per app.
â”œâ”€â”€ prompts/        Reusable AI prompt templates.
â”œâ”€â”€ architecture/   System diagrams and design decisions.
â”œâ”€â”€ datasets/       Sample data for offline testing.
```

---

## Conventions

- Language: Python 3.12
- Framework: Streamlit
- Style: snake_case for all Python identifiers
- Single-file apps preferred. All app logic in `app.py` unless the file exceeds ~400 lines.
- Supporting modules: `config.py`, `data_catalog.py`, `map_builder.py`, `ai_assistant.py`, `sample_layers.py`
- No external databases. Data comes from free APIs or local CSV/GeoJSON files.
- Environment variables loaded from `.env` via `python-dotenv`. Never hardcode keys.

---

## Deployment

This is the dev source folder. Do not push this folder to GitHub.
Deploy copy: `Desktop\code\ai-earth-observation-feedback\` â†’ https://github.com/ebrahim8332/ai-earth-observation
Live URL: https://eoil-explorer.streamlit.app

---

## Current App

`apps/01_eo_explorer/` â€” EO Explorer v1.1. Days 1-3. Live.

---

## Key Rules

- Never hardcode API keys.
- Never modify `.env` or `.env.example` when adding features. Document new keys only in `.env.example`.
- Keep `requirements.txt` updated when adding a new package.
- Update `AGENTS.md` and `CLAUDE.md` when app status changes.
- Follow the four-question framework: What is happening? Why? What changed? What next?

---

## Working Rules

- Plan before implementing any change beyond a single line.
- Explain every step in plain English. Alnoor is a beginner.
- No code comments that describe what the code does. Only comment when the WHY is non-obvious.
- Follow the writing style in `Desktop\code\style\anti-ai-writing-style.md`.
- Update this file when app status changes.
