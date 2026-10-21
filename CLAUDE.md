# CLAUDE.md

Prices an equity option with three models and their greeks, market data from
Yahoo. FastAPI backend, PostgreSQL, a page built in vanilla JS.

## Commands

```bash
source .venv/bin/activate

# Development server
uvicorn app.main:app --reload          # http://127.0.0.1:8000, /docs — Swagger

# Database schema
alembic upgrade head                                 # apply
alembic revision --autogenerate -m "what changed"    # after editing app/db/entities.py
alembic downgrade -1
alembic current

# Everything at once, in containers
docker compose up                       # app + PostgreSQL
docker compose down -v                  # stop and wipe the data

# Replacing the instrument table outright (not an endpoint, command line only)
python -m scripts.sync_instruments --replace
```

The database is created by hand, once: `psql -U postgres -h localhost -d
postgres -f sql/00_init_db.sql`. The connection string comes from `.env`; in
`alembic.ini` the `sqlalchemy.url` field is deliberately empty — `alembic/env.py`
fills it in from the settings.

## Environment Variables

`.env` in the root, `.env.example` as the template. In containers the values
are set in `compose.yml`.

- `DATABASE_URL` — PostgreSQL connection string, required
- `MARKET_CACHE_TTL_MINUTES` — how many minutes market data counts as fresh (15)
- `CALCULATION_RETENTION_DAYS` — how many days calculation history is kept (3)
- `MIN_QUOTED_STRIKES` — liquidity floor for an instrument to make the list (20)
- `DEFAULT_RISK_FREE_RATE` — fallback rate when `^IRX` is unavailable (0.042)
- `RATE_LIMIT` — requests per address, everything (120/minute)
- `PRICING_RATE_LIMIT` — pricing endpoints separately (30/minute)
- `MAX_CONCURRENT_PRICING` — simultaneous calculations, the memory ceiling (4)
- `ADMIN_TOKEN` — token for `POST /sync`; empty means "generate one at startup"

## Architecture

The path of one calculation: `static/app.js` → `routers/pricing.py` →
`services/pricing_service.py` → `pricing/<model>.py` → the answer is written
back through `db/repository.py`.

**The layers are separated strictly, and that is the project's main constraint.**

### Pricing (`app/pricing/`)

Pure math. Imports numpy, scipy and `arch` — no FastAPI, no database, no
yfinance. Debugs from a console, away from the browser.

- `models.py` — the contract every model obeys: `OptionParams` in,
  `PricingResult` out. `OptionType`, `ExerciseStyle`, `PricingModel`, and the
  `AmericanNotSupported` and `PriceOutOfBounds` exceptions. Sanity limits:
  `MAX_SIGMA = 5.0`, `MAX_RATE = 1.0`, `MIN_RATE = -0.5`,
  `MAX_DIV_YIELD = 5.0`
- `black_scholes.py` — `calculate`, `price`, `greeks`; analytic derivatives,
  with no repricing. The reference the other two models are measured against
- `binomial.py` — a CRR tree, `DEFAULT_STEPS = 500`, capped at 5000. Delta,
  gamma and theta are read off the first three layers, vega and rho
  numerically. The only model that handles the American style
- `monte_carlo.py` — GBM on the terminal price, antithetic pairs,
  `DEFAULT_PATHS = 100 000`, capped at 5 million, `SEED = 20240814`. A separate
  `estimate()` for the convergence chart — no greeks, twelve times cheaper
- `numeric_greeks.py` — central differences, clamped to the edges of the
  admissible range. Used by the tree, by Monte Carlo and by the tests
- `bounds.py` — no-arbitrage bounds. `require_within_bounds` raises,
  `violation` only reports (for Monte Carlo)
- `implied_vol.py` — `solve()` by Brent's method from scipy, with four
  intelligible reasons for failure instead of one "no solution"
- `parity.py` — put-call parity: `check`, `consensus_forward`,
  `implied_forward`, `implied_div_yield`. For the American style the bounds are
  computed on executable prices, not on mids
- `volatility.py` — `fit_garch`, `fit_egarch` through the `arch` library, and
  `historical`. What goes into a price is `term` (the average over the term),
  not `next_day`

### Services (`app/services/`)

- `market_data.py` — **the only place where `yfinance` and `pandas` live**. No
  DataFrame crosses the module boundary. `get_market_snapshot`,
  `get_option_chain`, `get_log_returns`, `fetch_screener_instruments`,
  `effective_div_yield`, plus the in-memory instrument registry
  (`set_instruments`, `prime_cache`). Changing the data provider must touch
  this file and no other
- `pricing_service.py` — **the only point where HTTP schemas and the math
  meet**: computes T, assembles `OptionParams`, dispatches on `PricingModel`.
  `calculate`, `compare`, `convergence`, `greek_profiles`, `option_chain`,
  `contract_quote`, `parity_report`, `volatility_report`
- `cache.py` — a `TTLCache` whose clock the caller owns. An expired entry is
  not dropped: if the source fails it is served with a note

### Core (`app/`)

- `main.py` — assembles the application. `lifespan` restores the registry and
  the cache from the database and starts two background tasks:
  `_purge_calculations_forever` (history cleanup, hourly) and
  `_bootstrap_instruments` (filling an empty instrument table). `index` serves
  the page as a string, substituting the token into a meta tag.
  `validation_handler` joins Pydantic's errors into a single Russian sentence
- `config.py` — `Settings` on pydantic-settings, reads `.env`
- `limits.py` — `limiter` (slowapi) and `pricing_slot()`, a per-event-loop
  semaphore
- `security.py` — `require_admin_token`, a FastAPI dependency; comparison
  through `compare_digest`
- `db/entities.py` — three tables, `db/repository.py` — the queries,
  `db/session.py` — the engine and the session factory

### Routers (`app/routers/`)

| Endpoint | Purpose |
|---|---|
| `POST /api/price` | price and greeks from one model, written to history |
| `POST /api/price/compare` | three models on the same inputs; one failing is a table row, not an error |
| `POST /api/price/convergence` | 48 tree builds and 10 simulations; not written to history |
| `POST /api/price/profiles` | delta and gamma against spot, theta against time |
| `GET /api/instruments` | the instrument list, from the in-memory registry |
| `POST /api/instruments/sync` | extend the table from the screener; **needs `X-Admin-Token`** |
| `GET /api/instruments/{t}/market` | a snapshot: spot, volatilities, dividends, rate, expiries |
| `GET /api/instruments/{t}/volatility` | historical, GARCH, EGARCH over the option's term |
| `GET /api/instruments/{t}/chain` | the chain with implied volatility |
| `GET /api/instruments/{t}/quote` | the market price of one contract |
| `GET /api/instruments/{t}/parity` | reconciling spot, rate and dividends against the market |
| `GET /api/health` | liveness check |

The text Swagger shows comes from the route's `summary=` and `description=`
parameters, not from the function's docstring.

### Frontend (`app/static/`)

Vanilla JS, four files plus `vendor/plotly-basic.min.js`.

- `index.html` — two columns: inputs on the left (sticky), the result and five
  tabs on the right
- `app.js` — loading data, running the calculation, drawing every chart through
  `Plotly.newPlot`, and `showTab`, the tab switcher
- `command.js` — the command line. Loaded after `app.js` and calls its
  functions directly; **it makes no API requests of its own** — it fills in the
  same fields and presses the same buttons as the mouse
- `style.css` — the dark terminal theme

## Database Tables

- `instruments` — `ticker` (primary key), `name`, `exchange`, `currency`,
  `has_options`, `quoted_strikes`, `created_at`, `synced_at`
- `market_snapshots` — `id`, `ticker` (FK with `ON DELETE CASCADE`),
  `fetched_at`, `as_of`, `spot`, `sigma_30d`, `sigma_90d`, `sigma_252d`,
  `div_yield`, `risk_free_rate`, `expirations` (JSONB), `div_schedule` (JSONB)
- `calculations` — `id`, `created_at`, the request parameters, `price`, the five
  greeks in columns of their own, `std_error`, `ci95_low`, `ci95_high`,
  `model_params` (JSONB), `elapsed_ms`

The greeks sit in separate columns rather than in one JSON: they have to be
sorted on and charted. JSONB takes only what has no common set of fields:
`steps` for the tree, `paths` for Monte Carlo.

## Conventions that are easy to break

- **Rates and volatilities are fractions of 1 everywhere** (0.0412, not 4.12).
  The conversion from percent is the frontend's job
- **252 against 365 is a deliberate difference.** Volatility is annualized
  through trading days; the term and the discounting through calendar days.
  Money is discounted every day, and a price does not move over a weekend
- **Greeks come back in terminal units:** vega and rho per percentage point,
  theta per calendar day. Each model does that conversion itself
- **The server computes T** from the expiry date: the browser may sit in
  another time zone
- **Validation is duplicated** in `schemas/pricing.py` and in
  `OptionParams.__post_init__` — not by oversight: the schema explains a typo
  at the HTTP boundary, the dataclass protects a library that can be called
  past the API
- **`GET /{ticker}/market` is a plain `def`, not `async`**: yfinance goes to
  the network synchronously. `POST /price` is `async`, but the calculation
  itself goes into `run_in_threadpool`
- **A failure to write history must not cost a computed price**: it is logged
  and the answer goes out with `calculation_id = null`
- **`history(auto_adjust=True)`** matters: without it splits and ex-dividend
  dates inflate historical volatility

### Data and the source

- **A closed exchange is detected by the maximum falling below the floor.**
  Outside trading hours Yahoo returns zero bid and ask; checking for "all
  zeros" is not enough — in a night measurement eleven instruments out of
  forty-three still had between one and nine strikes. By day the maximum was
  376, by night nine
- **`div_yield` is not the trailing yield.** It is computed for one specific
  expiry from the ex-dates: zero before the next one, a jump once one falls
  inside the term. The `div_yield` field in a snapshot stayed on for reference;
  the calculation takes `div_yield_by_expiry[expiry]`
- **The database stores the payment schedule `div_schedule`, not a ready q.** q
  depends on today's date and is wrong tomorrow, so it is recomputed in
  `market_data.prime_cache` every time the cache is raised
- **Not a single ticker is hard-coded.** The list comes from the Yahoo
  screener; on an empty database `main.py` fetches it itself, in a background
  task
- **Filling the instrument table does not block startup.** It used to be
  awaited before the `yield` in lifespan, and when Yahoo timed out uvicorn
  accepted no requests at all — neither the page nor `/api/health` answered for
  minutes

### Models and the calculation

- **Greek profiles are computed by the model the style implies, not the one the
  user picked.** For the European style, Black-Scholes (the exact value); for
  the American, a tree on 200 steps and 60 points. Monte Carlo will not do:
  sixty points times twelve simulations is tens of seconds, and the curve would
  come out ragged
- **There is deliberately no chart of vega against volatility.** For an
  at-the-money option vega barely depends on σ: 0.379 at 10% against 0.388 at
  80%. It would come out a horizontal line
- **The market price of a contract is a separate endpoint, not part of
  `/price`.** The chain takes 360 ms to fetch, the calculation single
  milliseconds; the page asks for both in parallel. The last trade price is not
  shown
- **Parity is computed on bid/ask, not on the last trade.** The breach bounds
  depend on our spot, the forward does not; the forward must not be rejected by
  those bounds, or a drifted spot throws away exactly the number that reveals
  the drift
- **Agreement in the reconciliation is decided on the forward, not on q.** q is
  an annual rate, and over a one-month term the same shift in the forward
  produces twelve times the discrepancy in q

### Load

- **Two different things are limited, and both are needed.** Frequency, by
  slowapi — `RATE_LIMIT` on everything and `PRICING_RATE_LIMIT` on
  calculations. Concurrency, by the `limits.pricing_slot()` semaphore wrapped
  around every `run_in_threadpool` in `routers/pricing.py`. A frequency limit
  does not hold memory down: "thirty a minute" does not stop all thirty
  arriving in the same second, and Monte Carlo on five million paths takes
  200 MB. The semaphore, in turn, does not refuse anything — it queues
- **The semaphore is created per event loop**, not once per module:
  `asyncio.Semaphore` binds to the first loop that touches it and fails in any
  other with "bound to a different event loop"

### Tests and migrations

- **Tests go to their own database, `<main>_test`.** The `use_test_database`
  fixture is autouse and session-scoped for a reason: when the swap was opt-in,
  fifty-odd tests running before the first one that asked for it managed to
  write history into the production database. Tables are recreated through
  `drop_all` + `create_all` — `create_all` does not alter existing ones, so
  after a schema change the test database would silently stay as it was
- **`pytest` must be green without PostgreSQL.** Five tests take the `database`
  fixture and skip with an explanation
- **The test database engine uses `NullPool`:** every `TestClient` has its own
  event loop, and a pool binds connections to the loop that opened them
- **`URL.render_as_string(hide_password=False)`, not `str(url)`:** `str()`
  puts "***" where the password should be
- **Never call `purge_old_calculations(retention_days=0)` in a test** — it means
  "delete everything", and it wiped the table once already
- **The limiter is off in tests**, through the `rate_limits_off` fixture, or the
  suite would run into its own protection
- **A new NOT NULL column on a populated table only goes through with a
  `server_default`**, which is dropped immediately afterwards by
  `alter_column(server_default=None)`
