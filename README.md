# Fano Explorer

An autonomous multi-agent research system for exploring the intersection of Fano-plane incidence geometry, Sanskrit grammar, and Indian music theory—guided by Sadhguru's teachings.

## Philosophy

The system operates on the principle that mathematical truth should feel **discovered, not invented**. Good findings are:
- **Natural** — they arise without forcing
- **Elegant** — minimal, symmetric, beautiful  
- **Inevitable** — once seen, they couldn't be otherwise
- **Decodifying** — they explain the specific numbers in the source teachings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AXIOM STORE                              │
│  Sadhguru excerpts, target numbers, blessed insights        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  Decides exploration direction, formulates prompts,         │
│  judges chunk-readiness, synthesizes write-ups              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ ChatGPT  │    │ Gemini   │    │ Claude   │
       │ Pro      │◄──►│ Deep     │    │ (backup) │
       │          │    │ Think    │    │          │
       └──────────┘    └──────────┘    └──────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    REVIEW QUEUE                              │
│  Chunks awaiting human judgment                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Human Oracle  │
                    │   ⚡  ?  ✗      │
                    └─────────────────┘
```

## Feedback Signals

| Signal | Meaning | System Response |
|--------|---------|-----------------|
| ⚡ **Profound** | "This is real" | → Blessed axioms, high exploration weight |
| ? **Interesting** | "Not sure yet" | → Incubation queue, revisit with fresh angles |
| ✗ **Wrong** | "Barren path" | → Negative weight, steer away |

## Setup

### Prerequisites
- Python 3.10+
- Chrome browser
- Google account (for ChatGPT/Gemini SSO)

### Installation

```bash
cd fano-explorer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### First Run (Authentication)

```bash
python fano_explorer.py auth
```

This opens Chrome and lets you log into ChatGPT and Gemini manually. Sessions are saved for future runs.

### Configuration

Edit `config.yaml` to customize:
- Polling intervals
- Rate limit handling
- Model preferences
- Exploration parameters

### Add Your Axioms

Place Sadhguru excerpts in `data/axioms/sadhguru_excerpts/` as markdown files.
Edit `data/axioms/target_numbers.yaml` with the specific numbers to decode.

## Usage

### Start Exploration

```bash
python fano_explorer.py start
```

The system runs continuously, exploring and generating chunks. Stop with `Ctrl+C`.

### Review Chunks

```bash
python fano_explorer.py review
```

Opens a local web interface at `http://localhost:8765` where you can review pending chunks and provide feedback.

### Check Status

```bash
python fano_explorer.py status
```

Shows current exploration threads, pending chunks, and rate limit status.

## Project Structure

```
fano-explorer/
├── fano_explorer.py          # Main entry point
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── src/
│   ├── orchestrator.py       # The brain
│   ├── browser/              # Browser automation
│   │   ├── base.py           # Common browser utilities
│   │   ├── chatgpt.py        # ChatGPT interface
│   │   └── gemini.py         # Gemini interface
│   ├── models/               # Data models
│   │   ├── thread.py         # Exploration threads
│   │   ├── chunk.py          # Research chunks
│   │   └── axiom.py          # Axioms and insights
│   ├── storage/              # Persistence
│   │   └── db.py             # SQLite operations
│   └── ui/                   # Human interface
│       └── review_server.py  # Local review web app
├── data/
│   ├── axioms/
│   │   ├── sadhguru_excerpts/  # Your source texts
│   │   ├── target_numbers.yaml # Numbers to decode
│   │   └── blessed_insights/   # ⚡ Profound findings
│   ├── explorations/           # Thread histories
│   └── chunks/
│       ├── pending/            # Awaiting review
│       ├── profound/           # ⚡ Blessed
│       ├── interesting/        # ? Incubating
│       └── rejected/           # ✗ Discarded
├── templates/
│   └── review.html             # Review interface
└── logs/
    └── exploration.log
```

## Rate Limit Handling

The system automatically detects rate limit messages from ChatGPT and Gemini:
- "You've reached the limit" → backs off exponentially
- "Deep Think unavailable" → switches to standard mode or waits
- "Try again tomorrow" → pauses that model until reset

Status is persisted, so you can stop and restart without losing rate limit state.

## License

Private research tool.
