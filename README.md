# Options Pricing

Prices an equity option with three models at once and shows how far the result
sits from the market. Market data is pulled from Yahoo live.

What it does:

- **three models** — Black-Scholes, a binomial tree, Monte Carlo — and the
  greeks for each: delta, gamma, vega, theta, rho
- **the market price of the same contract** next to the computed one, with a
  check of whether our price landed inside the spread
- **volatility of your choice**: historical, GARCH(1,1), EGARCH(1,1) —
  forecast over exactly the term of the option you picked
- **the volatility smile** across the whole chain of strikes
- **greek profiles**: how delta and gamma move with the share price, how theta
  accelerates towards expiry
- **model convergence**: how the tree and Monte Carlo close in on the exact
  value as steps and paths grow
- **market reconciliation** through put-call parity — it checks not the
  formulas but the spot, rate and dividends fed into them

There is a command line at the bottom of the page: the same work can be done
from the keyboard.

## Stack

**Frontend**

- Vanilla JavaScript
- Plotly (vendored, so the page works offline)
- CSS grid — two-column terminal dashboard with tabs
- A command line that drives the same endpoints as the form

**Backend**

- Python 3.12
- FastAPI
- PostgreSQL
- SQLAlchemy 2.0 (async) + Alembic
- NumPy, SciPy
- arch (GARCH, EGARCH)
- slowapi (rate limiting)

**External APIs**

- Yahoo Finance via yfinance

## Running it

Docker is all you need:

```bash
docker compose up
```

Then open http://localhost:8000

PostgreSQL comes up, the database is created, migrations run and the app
starts. Neither Python nor PostgreSQL has to be installed.

## Using it

The page is two columns: what you type on the left, what you read on the right.

**1. Pick a ticker** in the list on the left. Market data loads with it: the
spot price, historical volatility over three windows, the risk-free rate, the
next ex-dividend date and the available expiry dates.

**2. Check the option parameters.** The strike is filled in rounded to the
exchange grid, the expiry is the nearest monthly one. Switch call/put and the
exercise style if you need to.

The dividend yield is recomputed whenever the expiry changes.

**3. Choose a volatility** — press "Подобрать модели" in the "Волатильность:
чем считать" card. A table appears: historical over a year, the GARCH forecast
and the EGARCH forecast, both over exactly your option's term. Press "выбрать"
in the row you want and the value goes into the form.

**4. Press "Рассчитать".** On the right you get the model price and, next to
it, the market price of that contract if it trades. The greeks are underneath.

What matters in that comparison is not the gap but **whether the computed price
landed inside the spread**. If it did, the market agrees with you within its
own uncertainty. If it did not, the page says which way it missed and why that
usually happens.

### The tabs on the right

| tab | what it shows |
|---|---|
| **Сравнение** | all three models on the same inputs, in one table |
| **Профили греков** | delta and gamma against the share price, theta against time; the dot on the curve is your own parameters |
| **Сходимость** | how the tree and Monte Carlo approach the exact value |
| **Улыбка** | implied volatility across every strike in the chain |
| **Сверка с рынком** | put-call parity: whether our spot, rate and dividends agree with the market's |

### The command line

An input line sits at the bottom. Commands trigger the same calculations as the
buttons:

```
use AAPL                        pick a ticker
price                           recompute whatever is in the form
price 340 2026-11-20 call       price one specific contract
price --model mc --sigma 24.5   another model, another σ
compare                         all three models at once
vol                             the volatility table
smile  greeks  conv  parity     open the matching tab
help                            list the commands
clear                           clear the log
```

Arguments are optional: without them a command takes what is already in the
form. History is on the up and down arrows.

## Running it without Docker

Python 3.12 and PostgreSQL required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # put in the password of your own PostgreSQL

psql -U postgres -h localhost -d postgres -f sql/00_init_db.sql
alembic upgrade head

uvicorn app.main:app --reload
```

Without PostgreSQL the app will start but will be of no use: the ticker list
stays empty and syncing returns an error — there is nowhere to keep the
instrument table.

## Tests

```bash
pytest
```

## API

Interactive documentation, with a button to run each request —
http://localhost:8000/docs

```
POST /api/price                            price and greeks, one model
POST /api/price/compare                    all three models
POST /api/price/convergence                convergence of the tree and Monte Carlo
POST /api/price/profiles                   greek profiles

GET  /api/instruments                      the ticker list
GET  /api/instruments/{ticker}/market      spot, volatility, dividends, rate
GET  /api/instruments/{ticker}/volatility  historical, GARCH, EGARCH
GET  /api/instruments/{ticker}/chain       option chain with implied σ
GET  /api/instruments/{ticker}/quote       market price of one contract
GET  /api/instruments/{ticker}/parity      put-call parity reconciliation
POST /api/instruments/sync                 refresh the ticker list (token needed)
```

Calculation history is written to the `calculations` table but is never served
over HTTP: you are meant to read it in PostgreSQL directly. Rows older than
three days are removed by a background task.

## Configuration

Everything lives in `.env`, see `.env.example`. When running through Docker the
values needed are already set in `compose.yml`.

| setting | default | what it does |
|---|---|---|
| `DATABASE_URL` | — | where PostgreSQL is |
| `MARKET_CACHE_TTL_MINUTES` | 15 | how long market data counts as fresh |
| `CALCULATION_RETENTION_DAYS` | 3 | how long calculation history is kept |
| `MIN_QUOTED_STRIKES` | 20 | liquidity floor for making the list |
| `DEFAULT_RISK_FREE_RATE` | 0.042 | used when Yahoo returns no rate |
| `RATE_LIMIT` | 120/min | requests from one address |
| `PRICING_RATE_LIMIT` | 30/min | a separate cap on calculations |
| `MAX_CONCURRENT_PRICING` | 4 | calculations at the same time |
| `ADMIN_TOKEN` | generated | token for `POST /sync` |

The schema is kept by Alembic migrations:

```bash
alembic upgrade head     # bring the database to the current schema
alembic current          # which version the database is on
alembic downgrade -1     # roll the last one back
```

The connection string comes from `.env` and is not in `alembic.ini`, so the
password never reaches the repository.

## Admin commands

Replacing the instrument table outright — it deletes whatever is missing from
the screener's new answer, together with their market snapshots:

```bash
python -m scripts.sync_instruments --replace
```

It prints the deletion list and waits for confirmation. Doing this over HTTP is
deliberately impossible: `POST /api/instruments/sync` only adds and refreshes.

## Layout

```
app/
├── pricing/    the math: three models, greeks, arbitrage bounds,
│               implied volatility, parity, GARCH
├── services/   market data (the only module that knows yfinance)
│               and calculation orchestration
├── db/         SQLAlchemy async
├── schemas/    Pydantic API contracts
├── routers/    the HTTP layer
└── static/     the page: HTML, CSS, app.js, command.js
```
