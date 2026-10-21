# Options Pricing

Prices equity options with three models at once and shows how far each result
sits from the live market. Market data comes from Yahoo Finance in real time.

The interface is a terminal-style dashboard with a working command line.

## What it does

- Price European and American options with Black-Scholes, a CRR binomial tree,
  and Monte Carlo
- Compute all five greeks for every model
- Show the live market price of the same contract next to the model price
- Pick the volatility input: realized, GARCH(1,1), or EGARCH(1,1), forecast over
  the option's own term
- Plot the volatility smile across the whole option chain
- Plot greek profiles: delta and gamma against spot, theta against time
- Plot model convergence: how the tree and Monte Carlo approach the exact value
- Reconcile inputs against the market through put-call parity
- Drive all of it from a command line instead of the form

## Why this project is interesting

The hard part of option pricing is not the formula — it is knowing whether the
numbers you feed it are right. This project separates those two questions:

- **Internal checks confirm the math.** The tree and Monte Carlo converge to
  Black-Scholes, numeric greeks match the analytic ones, and every price is
  verified against no-arbitrage bounds — the one check that catches a mistake in
  a formula rather than in a form.
- **External checks confirm the inputs.** Put-call parity requires
  `C − P = S·e^(−qT) − K·e^(−rT)`, where the left side is pure market prices and
  the right side is our spot, rate, and dividend yield. Volatility does not
  appear, so the check cannot collapse into a tautology.
- **Dividends are discrete, not a constant yield.** `q` is computed per expiry
  from actual ex-dates: zero before the next one, then a jump. A quarterly
  dividend landing inside a two-week option is over 100% annualized, and that is
  the correct number, not an error.
- **Transient failures are never stored as facts.** A timed-out option check is
  recorded as "unknown", not "no options". A cancelled dividend needs two
  independent signals. A closed market leaves liquidity unmeasured rather than
  writing zero.
- **Layers are enforced.** `app/pricing/` imports numpy and scipy and nothing
  else — no web framework, no database, no data provider. The math debugs from a
  console, and 698 tests run in four seconds because none of them start a server
  or open a socket.

## Tech stack

**Frontend**

- Vanilla JavaScript, no build step
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

- Yahoo Finance via yfinance — prices, dividends, option chains, risk-free rate

## Running locally

### With Docker

```bash
docker compose up
```

Then open http://localhost:8000

PostgreSQL starts alongside, migrations run, and the ticker list fills in the
background. No Python or PostgreSQL installation, no `.env`.

### Without Docker

Requires Python 3.12 and PostgreSQL.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add your PostgreSQL password

psql -U postgres -h localhost -d postgres -f sql/00_init_db.sql
alembic upgrade head

uvicorn app.main:app --reload
```

## Using it

1. **Pick a ticker** on the left. Spot, realized volatility, the risk-free rate
   and the next ex-dividend date load automatically.
2. **Check the option parameters.** The strike snaps to the exchange grid. The
   dividend yield changes when you change the expiry — that is not a bug.
3. **Choose a volatility.** Press "Подобрать модели" to compare realized, GARCH
   and EGARCH, each forecast over your option's own term, then pick one.
4. **Press "Рассчитать".** The model price appears next to the live market
   price. What matters is not the gap but whether the model price lands inside
   the bid-ask spread.

Five tabs on the right hold model comparison, greek profiles, convergence, the
volatility smile, and the market reconciliation.

The command line at the bottom does the same work from the keyboard:

```
use AAPL                        pick a ticker
price 340 2026-11-06 call       price one contract
price --model mc --sigma 24.5   same contract, different model and sigma
compare                         all three models
vol                             the volatility table
smile  greeks  conv  parity     open the matching tab
help                            list commands
```

Arguments are optional: without them a command uses whatever is in the form.

## Outside market hours

The US market trades 09:30–16:00 New York time. Outside that window Yahoo
returns zero bid and ask on every strike, so anything priced off live quotes is
unavailable: the option chain, the volatility smile, the market reconciliation
and the contract quote. The app says so explicitly instead of failing.

Everything else works around the clock — ticker selection, market data, realized
volatility, GARCH and EGARCH, all three pricing models, greeks, greek profiles
and convergence.

## Tests

```bash
pytest
```

698 tests, about four seconds. No network: the data source is monkeypatched and
the math is called directly.

PostgreSQL is optional — the five tests that read tables skip with an
explanation. When a database is present, tests create their own `pricing_test`
and write only there.

```bash
pip install ruff
ruff check .
```

Both run automatically on every push, on a clean machine with PostgreSQL
attached. See `.github/workflows/tests.yml`.

## API

Interactive docs at http://localhost:8000/docs

```
POST /api/price                            price and greeks, one model
POST /api/price/compare                    all three models
POST /api/price/convergence                tree and Monte Carlo convergence
POST /api/price/profiles                   greek profiles

GET  /api/instruments                      ticker list
GET  /api/instruments/{ticker}/market      spot, volatility, dividends, rate
GET  /api/instruments/{ticker}/volatility  realized, GARCH, EGARCH
GET  /api/instruments/{ticker}/chain       option chain with implied vol
GET  /api/instruments/{ticker}/quote       live price of one contract
GET  /api/instruments/{ticker}/parity      put-call parity reconciliation
POST /api/instruments/sync                 refresh the ticker list (token)
```

Calculation history is written to the `calculations` table but never served over
HTTP — read it in PostgreSQL directly. Rows older than three days are removed by
a background task.

## Configuration

Everything lives in `.env`, see `.env.example`. Docker supplies its own values
in `compose.yml`.

| Setting | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `MARKET_CACHE_TTL_MINUTES` | 15 | how long market data counts as fresh |
| `CALCULATION_RETENTION_DAYS` | 3 | how long calculation history is kept |
| `MIN_QUOTED_STRIKES` | 20 | liquidity floor for the ticker list |
| `DEFAULT_RISK_FREE_RATE` | 0.042 | fallback when Yahoo has no rate |
| `RATE_LIMIT` | 120/minute | requests per address |
| `PRICING_RATE_LIMIT` | 30/minute | separate cap on pricing endpoints |
| `MAX_CONCURRENT_PRICING` | 4 | simultaneous calculations |
| `ADMIN_TOKEN` | generated | token for `POST /sync` |

The last three bound load. Two rate limits plus a separate concurrency cap:
Monte Carlo at five million paths holds 200 MB, and "thirty per minute" would
not stop all thirty from arriving in the same second.

Replacing the ticker list outright is a command, not an endpoint — it deletes
instruments and cascades to their market snapshots:

```bash
python -m scripts.sync_instruments --replace
```

## Project structure

```
app/
  pricing/     three models, greeks, arbitrage bounds,
               implied volatility, put-call parity, GARCH
  services/    market data (the only module that knows yfinance)
               and calculation orchestration
  db/          SQLAlchemy async
  schemas/     Pydantic API contracts
  routers/     HTTP layer
  static/      the page: HTML, CSS, app.js, command.js

alembic/       schema migrations
scripts/       admin commands
tests/         698 tests
```

`app/pricing/` knows nothing about the network or the database: option
parameters in, price and greeks out. That is why the math debugs from a console
and the whole suite runs in four seconds.
