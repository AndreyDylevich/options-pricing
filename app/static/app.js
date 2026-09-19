// Логика страницы: выбор акции, форма параметров, три кнопки моделей,
// вывод цены и греков.

const select = document.getElementById("instrument");
const instrumentStatus = document.getElementById("instrument-status");
const marketCard = document.getElementById("market-card");
const marketNote = document.getElementById("market-note");
const paramsCard = document.getElementById("params-card");
const paramsForm = document.getElementById("params");
const paramsStatus = document.getElementById("params-status");
const calcCard = document.getElementById("calc-card");
const calcStatus = document.getElementById("calc-status");
const resultCard = document.getElementById("result-card");
const compareCard = document.getElementById("compare-card");
const compareNote = document.getElementById("compare-note");
const chainCard = document.getElementById("chain-card");
const chainNote = document.getElementById("chain-note");
const convergenceCard = document.getElementById("convergence-card");
const convergenceNote = document.getElementById("convergence-note");
const parityCard = document.getElementById("parity-card");
const parityNote = document.getElementById("parity-note");
const paritySummary = document.getElementById("parity-summary");

const MODEL_NAMES = {
  black_scholes: "Блэк-Шоулз",
  binomial: "Биномиальное дерево",
  monte_carlo: "Монте-Карло",
};

// Какой моделью считать. Меняется кнопками, запускается «Рассчитать».
let selectedModel = "black_scholes";


// Последний загруженный снимок рынка. Из него заполняется форма, и по
// нему же работает кнопка «Вернуть рыночные значения».
let snapshot = null;

// Дней в году для перевода срока до экспирации в годы. Календарные, а не
// торговые: деньги дисконтируются каждый день, включая выходные.
const DAYS_IN_YEAR = 365;


// --- список акций ---------------------------------------------------

async function loadInstruments() {
  try {
    const instruments = await getJSON("/api/instruments");
    fillSelect(instruments);
  } catch (error) {
    select.innerHTML = '<option value="">—</option>';
    setStatus(instrumentStatus, `Не удалось загрузить список: ${error.message}`, true);
  }
}


function fillSelect(instruments) {
  select.innerHTML = '<option value="">— выберите акцию —</option>';

  for (const item of instruments) {
    const option = document.createElement("option");
    option.value = item.ticker;
    option.textContent = `${item.ticker} — ${item.name}`;
    select.append(option);
  }

  select.disabled = false;
  setStatus(instrumentStatus, `Доступно инструментов: ${instruments.length}`);
}


/** Подтянуть список бумаг из скринера Yahoo. */
async function syncInstruments() {
  const button = document.getElementById("sync-instruments");
  button.disabled = true;
  setStatus(instrumentStatus, "Спрашиваем скринер и проверяем опционы…");

  try {
    const summary = await postJSON("/api/instruments/sync", {}, { admin: true });
    await loadInstruments();
    setStatus(instrumentStatus,
      `Список обновлён: получено ${summary.fetched}, `
      + `новых ${summary.added}, обновлено ${summary.updated}. `
      + `Доступно для расчёта: ${summary.available}`);
  } catch (error) {
    setStatus(instrumentStatus, `Не удалось обновить список: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
}


// --- рыночные данные ------------------------------------------------

async function loadMarket(ticker, refresh = false) {
  marketCard.hidden = false;
  paramsCard.hidden = true;
  setStatus(marketNote, refresh ? "Обновляем с рынка…" : "Запрашиваем данные…");
  clearMetrics();

  try {
    const query = refresh ? "?refresh=true" : "";
    snapshot = await getJSON(`/api/instruments/${ticker}/market${query}`);
    renderMarket(snapshot);
    fillParams(snapshot);
  } catch (error) {
    snapshot = null;
    setStatus(marketNote, error.message, true);
  }
}


function renderMarket(data) {
  setText("m-spot", formatMoney(data.spot, data.currency));
  setText("m-sigma30", formatPercent(data.sigma_30d));
  setText("m-sigma90", formatPercent(data.sigma_90d));
  setText("m-sigma252", formatPercent(data.sigma_252d));
  setText("m-div", formatPercent(data.div_yield));
  setText("m-exdate", nextExDate(data));
  setText("m-rate", formatPercent(data.risk_free_rate));
  setText("m-expirations", String(data.expirations.length));

  const parts = [`Данные на ${formatDate(data.as_of)}`];

  if (data.expirations.length === 0) {
    parts.push("опционы по этой бумаге недоступны");
  } else {
    parts.push(`ближайшая экспирация ${data.expirations[0]}`);
  }

  parts.push(cacheState(data));
  setStatus(marketNote, parts.join(" · "), data.stale);
}


/** Ближайшая дивидендная отсечка: с неё скачком меняется q. */
function nextExDate(data) {
  const next = (data.div_schedule ?? [])[0];

  if (!next) {
    return "дивидендов нет";
  }

  const when = formatDay(next.ex_date);
  const amount = formatMoney(next.amount, data.currency);

  return next.estimated ? `${when} · ${amount} (прогноз)` : `${when} · ${amount}`;
}


/** Откуда взялись показанные цифры: с рынка, из кэша или из старого кэша. */
function cacheState(data) {
  const minutes = Math.round((Date.now() - new Date(data.fetched_at)) / 60000);
  const ago = minutes < 1 ? "только что" : `${minutes} мин назад`;

  if (data.stale) {
    return `источник недоступен, показаны данные от ${ago} — нажмите «Обновить с рынка»`;
  }
  if (data.cached) {
    return `из кэша, запрошено ${ago}`;
  }
  return "запрошено с рынка только что";
}


function clearMetrics() {
  for (const id of ["m-spot", "m-sigma30", "m-sigma90", "m-sigma252",
                    "m-div", "m-exdate", "m-rate", "m-expirations"]) {
    setText(id, "—");
  }
}


// --- форма параметров -----------------------------------------------

function fillParams(data) {
  if (data.expirations.length === 0) {
    paramsCard.hidden = true;
    return;
  }

  setValue("p-spot", round(data.spot, 2));
  setValue("p-strike", nearestStrike(data.spot));

  fillExpirations(data.expirations);

  // Волатильность по умолчанию — 30-дневная. Если истории не хватило,
  // спускаемся к более длинным окнам, лишь бы поле не осталось пустым.
  const sigma = data.sigma_30d ?? data.sigma_90d ?? data.sigma_252d;
  setValue("p-sigma", sigma === null ? "" : round(sigma * 100, 2));

  setValue("p-rate", round(data.risk_free_rate * 100, 2));
  applyDividendYield();

  paramsCard.hidden = false;
  calcCard.hidden = false;

  // Карточку показываем сразу — иначе кнопка «Загрузить цепочку», которая
  // внутри неё, была бы недоступна. Данные подтягиваются по нажатию:
  // цепочка это отдельный запрос в сеть на пару секунд.
  chainCard.hidden = false;
  parityCard.hidden = false;
  paritySummary.hidden = true;
  setStatus(parityNote,
    "Нажмите «Сверить», чтобы проверить наши спот, ставку и дивиденды "
    + "по рыночным ценам опционов");
  convergenceCard.hidden = false;
  document.getElementById("conv-tree").hidden = true;
  document.getElementById("conv-mc").hidden = true;
  setStatus(convergenceNote,
    "Нажмите «Построить сходимость», чтобы увидеть, как приближённые "
    + "модели подходят к точному значению");
  document.querySelector("#chain-table tbody").innerHTML = "";
  setStatus(chainNote,
    "Нажмите «Загрузить цепочку», чтобы посчитать волатильность "
    + "по рыночным ценам опционов");
  refreshHints();
}


function fillExpirations(expirations) {
  const field = document.getElementById("p-expiry");
  field.innerHTML = "";

  for (const value of expirations) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = `${value} · через ${daysUntil(value)} дн.`;
    field.append(option);
  }

  // По умолчанию — примерно месячный опцион: самый ходовой срок и при
  // этом достаточно далёкий, чтобы модели вели себя прилично.
  const monthly = expirations.findIndex((value) => daysUntil(value) >= 30);
  field.selectedIndex = monthly === -1 ? expirations.length - 1 : monthly;
}


/** Ближайший «круглый» страйк к текущей цене: опцион на деньгах. */
function nearestStrike(spot) {
  const step = spot >= 200 ? 5 : spot >= 50 ? 2.5 : 1;
  return round(Math.round(spot / step) * step, 2);
}


function daysUntil(isoDate) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const target = new Date(`${isoDate}T00:00:00`);
  return Math.round((target - today) / 86400000);
}


/**
 * Значения формы в том виде, в каком их ждёт API: проценты переведены в
 * доли единицы, срок до экспирации — в годы. Используется на этапе 4.
 */
function readParams() {
  const expiry = document.getElementById("p-expiry").value;

  return {
    ticker: select.value,
    spot: numberOf("p-spot"),
    strike: numberOf("p-strike"),
    expiry: expiry,
    time_to_expiry: daysUntil(expiry) / DAYS_IN_YEAR,
    option_type: checkedValue("option_type"),
    exercise: checkedValue("exercise"),
    sigma: numberOf("p-sigma") / 100,
    rate: numberOf("p-rate") / 100,
    div_yield: numberOf("p-div") / 100,
    steps: numberOf("p-steps"),
    paths: numberOf("p-paths"),
    market_as_of: snapshot?.as_of ?? null,
  };
}


/** Подсказки, которые пересчитываются при любой правке формы. */
/** Подставить q под выбранную экспирацию и объяснить, откуда оно взялось.
 *
 * Дивиденд платится скачком, а модели просят непрерывную ставку, поэтому
 * одного q на все сроки не бывает: опцион, экспирирующийся до ближайшей
 * отсечки, дивиденда не видит вовсе. Пересчёт делает сервер, здесь только
 * выбор нужного числа из карты.
 */
function applyDividendYield() {
  if (!snapshot) {
    return;
  }

  const expiry = document.getElementById("p-expiry").value;
  const schedule = snapshot.div_schedule ?? [];

  // Снимок, поднятый из базы после перезапуска, карты не содержит —
  // откатываемся на среднегодовое значение.
  const value = snapshot.div_yield_by_expiry?.[expiry] ?? snapshot.div_yield;
  setValue("p-div", round(value * 100, 2));

  const inside = schedule.filter((payment) => payment.ex_date <= expiry);
  setText("h-div", dividendHint(inside, snapshot.currency));
}


function dividendHint(inside, currency) {
  if (inside.length === 0) {
    return "до этой экспирации отсечек нет, q = 0";
  }

  const word = inside.length === 1 ? "отсечка"
    : inside.length < 5 ? "отсечки" : "отсечек";
  const total = inside.reduce((sum, payment) => sum + payment.amount, 0);
  const guessed = inside.some((payment) => payment.estimated)
    ? ", дальние даты — прогноз" : "";

  return `внутри срока ${inside.length} ${word} на ${formatMoney(total, currency)}`
    + ` — пересчитано в годовую ставку${guessed}`;
}


function refreshHints() {
  const expiry = document.getElementById("p-expiry").value;
  const days = daysUntil(expiry);
  const years = days / DAYS_IN_YEAR;
  setText("h-term", `T = ${days} дн. = ${years.toFixed(4)} года`);

  const spot = numberOf("p-spot");
  const strike = numberOf("p-strike");
  const isCall = checkedValue("option_type") === "call";
  setText("h-moneyness", moneyness(spot, strike, isCall));

  const american = checkedValue("exercise") === "american";
  setText("h-exercise", american
    ? "исполнение в любой день до экспирации — считает только дерево"
    : "исполнение только в дату экспирации");

  syncModelsWithExercise();
  validate();
}


/** На деньгах, в деньгах или вне денег — и насколько. */
function moneyness(spot, strike, isCall) {
  if (!spot || !strike) {
    return "—";
  }

  const diff = (spot - strike) / strike;

  if (Math.abs(diff) < 0.005) {
    return "на деньгах (ATM)";
  }

  const inTheMoney = isCall ? spot > strike : spot < strike;
  const label = inTheMoney ? "в деньгах (ITM)" : "вне денег (OTM)";
  return `${label}, S/K − 1 = ${(diff * 100).toFixed(1)}%`;
}


// Границы повторяют те, что стоят в схеме запроса и в OptionParams.
// Дублирование намеренное: страница объясняет ошибку сразу, не гоняя
// пользователя на сервер за очевидным.
const LIMITS = {
  sigma: { min: 0, max: 500 },
  rate: { min: -50, max: 100 },
  div: { min: 0, max: 100 },
  steps: { min: 1, max: 5000 },
  paths: { min: 1000, max: 5000000 },
  strikeRatio: { min: 0.01, max: 100 },
};


/** Проверка формы. Возвращает false и подсвечивает первую проблему. */
function validate() {
  const spot = numberOf("p-spot");
  const strike = numberOf("p-strike");
  const sigma = numberOf("p-sigma");
  const rate = numberOf("p-rate");
  const div = numberOf("p-div");
  const steps = numberOf("p-steps");
  const paths = numberOf("p-paths");

  const problems = [];

  if (!(spot > 0)) problems.push("цена должна быть больше нуля");
  if (!(strike > 0)) problems.push("страйк должен быть больше нуля");

  if (spot > 0 && strike > 0) {
    const ratio = strike / spot;
    if (ratio < LIMITS.strikeRatio.min || ratio > LIMITS.strikeRatio.max) {
      problems.push("страйк отличается от цены более чем в сто раз");
    }
  }

  if (!(sigma > 0)) {
    problems.push("волатильность должна быть больше нуля");
  } else if (sigma > LIMITS.sigma.max) {
    problems.push(`волатильность больше ${LIMITS.sigma.max}% — похоже на опечатку`);
  }

  if (rate < LIMITS.rate.min || rate > LIMITS.rate.max) {
    problems.push(`ставка вне диапазона ${LIMITS.rate.min}…${LIMITS.rate.max}%`);
  }
  if (div < LIMITS.div.min || div >= LIMITS.div.max) {
    problems.push(`дивидендная доходность вне диапазона 0…${LIMITS.div.max}%`);
  }

  if (!Number.isInteger(steps) || steps < LIMITS.steps.min || steps > LIMITS.steps.max) {
    problems.push(`шагов дерева: целое от ${LIMITS.steps.min} до ${LIMITS.steps.max}`);
  }
  if (!Number.isInteger(paths) || paths < LIMITS.paths.min || paths > LIMITS.paths.max) {
    problems.push(`путей Монте-Карло: целое от ${LIMITS.paths.min} до ${LIMITS.paths.max}`);
  }

  if (!document.getElementById("p-expiry").value) {
    problems.push("выберите дату экспирации");
  }

  if (problems.length) {
    setStatus(paramsStatus, `Проверьте параметры: ${problems.join(", ")}`, true);
    return false;
  }

  setStatus(paramsStatus, "Параметры заполнены");
  return true;
}


/** Американский стиль умеет только дерево — не даём выбрать остальные. */
function syncModelsWithExercise() {
  const american = checkedValue("exercise") === "american";

  for (const button of document.querySelectorAll(".btn[data-model]")) {
    const treeOnly = button.dataset.model !== "binomial";
    button.disabled = american && treeOnly;
    button.title = american && treeOnly
      ? "Американский стиль умеет считать только биномиальное дерево"
      : "";
  }

  if (american && selectedModel !== "binomial") {
    selectModel("binomial");
    setStatus(calcStatus, "Для американского стиля выбрано биномиальное дерево");
  }
}


// --- расчёт ---------------------------------------------------------

/** Выбор модели: кнопки работают как переключатель, а не как запуск. */
function selectModel(model) {
  selectedModel = model;

  for (const button of document.querySelectorAll(".btn[data-model]")) {
    button.classList.toggle("btn--active", button.dataset.model === model);
  }

  // Результат относится к прежней модели — прячем, чтобы не выглядело
  // так, будто это посчитала вновь выбранная.
  resultCard.hidden = true;
  compareCard.hidden = true;
}


function setButtonsDisabled(disabled) {
  document.getElementById("calculate").disabled = disabled;
}


async function runModel() {
  if (!validate()) {
    setStatus(calcStatus, "Сначала исправьте параметры выше", true);
    return;
  }

  setButtonsDisabled(true);
  setStatus(calcStatus, `${MODEL_NAMES[selectedModel]}: считаем…`);

  try {
    const result = await postJSON("/api/price", {
      ...readParams(),
      model: selectedModel,
    });
    renderResult(result);
    compareCard.hidden = true;
    setStatus(calcStatus, result.calculation_id === null
      ? "Расчёт выполнен, но сохранить в историю не удалось"
      : "");
  } catch (error) {
    resultCard.hidden = true;
    setStatus(calcStatus, error.message, true);
  } finally {
    setButtonsDisabled(false);
  }
}


function renderResult(result) {
  const meta = result.meta;
  const currency = snapshot?.currency ?? "USD";

  renderWarning(meta.warning);

  setText("r-model", MODEL_NAMES[meta.model] ?? meta.model);
  setText("r-price", formatMoney(result.price, currency));

  // У Монте-Карло цена — оценка, и без интервала она вводит в
  // заблуждение: показываем разброс прямо под ценой.
  setText("r-note", meta.ci95
    ? `95% интервал: ${meta.ci95[0].toFixed(4)} … ${meta.ci95[1].toFixed(4)} · `
      + `стандартная ошибка ${meta.std_error.toFixed(4)}`
    : "точное значение модели, случайной ошибки нет");

  setText("g-delta", result.greeks.delta.toFixed(4));
  setText("g-gamma", result.greeks.gamma.toFixed(4));
  setText("g-vega", result.greeks.vega.toFixed(4));
  setText("g-theta", result.greeks.theta.toFixed(4));
  setText("g-rho", result.greeks.rho.toFixed(4));

  setText("r-meta",
    `${meta.exercise === "american" ? "Американский" : "Европейский"} стиль · `
    + `T = ${meta.days_to_expiry} дн. (${meta.time_to_expiry.toFixed(4)} года) · `
    + `расчёт ${meta.elapsed_ms.toFixed(1)} мс`);

  resultCard.hidden = false;
}


/** Предупреждение модели: расчёт прошёл, но верить ему нельзя. */
function renderWarning(warning) {
  const box = document.getElementById("r-warning");
  box.hidden = !warning;
  box.textContent = warning ?? "";
}


// --- сравнение моделей ----------------------------------------------

async function runComparison() {
  if (!validate()) {
    setStatus(calcStatus, "Сначала исправьте параметры выше", true);
    return;
  }

  setButtonsDisabled(true);
  setStatus(calcStatus, "Считаем всеми тремя моделями…");

  // Модель в теле не нужна: сравнение считает все.
  const payload = readParams();
  delete payload.model;

  try {
    const comparison = await postJSON("/api/price/compare", payload);
    renderComparison(comparison);
    // Отдельный результат относится к одной модели — прячем, чтобы на
    // экране не было двух разных ответов на один вопрос.
    resultCard.hidden = true;
    setStatus(calcStatus, "");
  } catch (error) {
    compareCard.hidden = true;
    setStatus(calcStatus, error.message, true);
  } finally {
    setButtonsDisabled(false);
  }
}


function renderComparison(comparison) {
  const body = document.querySelector("#compare-table tbody");
  body.innerHTML = "";

  const priced = comparison.results.filter((item) => item.price !== null);

  // За точку отсчёта берём Блэка-Шоулза: это единственная модель с точной
  // формулой, остальные две к ней приближаются.
  const reference = priced.find((item) => item.model === "black_scholes")
    ?? priced[0];

  for (const item of comparison.results) {
    const row = document.createElement("tr");

    if (item.price === null) {
      row.className = "unavailable";
      appendCell(row, MODEL_NAMES[item.model] ?? item.model);
      const cell = document.createElement("td");
      cell.colSpan = 8;
      cell.textContent = item.error;
      row.append(cell);
      body.append(row);
      continue;
    }

    const isReference = reference && item.model === reference.model;
    const gap = isReference ? "эталон" : formatGap(item.price - reference.price);

    appendCell(row, MODEL_NAMES[item.model] ?? item.model);
    appendCell(row, item.price.toFixed(4), "num");
    appendCell(row, gap, isReference ? "num reference" : "num");
    appendCell(row, item.greeks.delta.toFixed(4), "num");
    appendCell(row, item.greeks.gamma.toFixed(4), "num");
    appendCell(row, item.greeks.vega.toFixed(4), "num");
    appendCell(row, item.greeks.theta.toFixed(4), "num");
    appendCell(row, item.greeks.rho.toFixed(4), "num");
    appendCell(row, item.elapsed_ms.toFixed(2), "num");

    body.append(row);
  }

  compareCard.hidden = false;
  setStatus(compareNote, comparisonNote(comparison, priced));
}


function appendCell(row, text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) {
    cell.className = className;
  }
  row.append(cell);
}


function formatGap(difference) {
  const sign = difference >= 0 ? "+" : "−";
  return `${sign}${Math.abs(difference).toFixed(4)}`;
}


/** Пояснение под таблицей: разброс цен и предупреждения моделей. */
function comparisonNote(comparison, priced) {
  const parts = [
    `${comparison.exercise === "american" ? "Американский" : "Европейский"} стиль`,
    `T = ${comparison.days_to_expiry} дн.`,
  ];

  if (priced.length > 1) {
    const prices = priced.map((item) => item.price);
    const spread = Math.max(...prices) - Math.min(...prices);
    parts.push(`разброс цен ${spread.toFixed(4)}`);
  }

  const monteCarlo = priced.find((item) => item.model === "monte_carlo");
  if (monteCarlo?.ci95) {
    parts.push(
      `Монте-Карло: 95% интервал ${monteCarlo.ci95[0].toFixed(4)} … `
      + `${monteCarlo.ci95[1].toFixed(4)}`
    );
  }

  const warnings = priced.filter((item) => item.warning).map((item) => item.warning);
  return [parts.join(" · "), ...warnings].join(" | ");
}


// --- подразумеваемая волатильность ----------------------------------

async function loadChain() {
  const ticker = select.value;
  const expiry = document.getElementById("p-expiry").value;

  if (!ticker || !expiry) {
    return;
  }

  const button = document.getElementById("load-chain");
  button.disabled = true;
  setStatus(chainNote, `Загружаем цепочку на ${expiry}…`);
  document.querySelector("#chain-table tbody").innerHTML = "";

  try {
    const chain = await getJSON(
      `/api/instruments/${ticker}/chain?expiry=${expiry}`
    );
    renderSmile(chain);
    renderChain(chain);
  } catch (error) {
    setStatus(chainNote, error.message, true);
  } finally {
    button.disabled = false;
  }
}


function renderChain(chain) {
  const body = document.querySelector("#chain-table tbody");
  body.innerHTML = "";

  // Ближе к деньгам — надёжнее: там спред узкий, а vega велика.
  const sorted = [...chain.quotes].sort(
    (a, b) => Math.abs(a.log_moneyness) - Math.abs(b.log_moneyness)
  );

  for (const quote of sorted) {
    const row = document.createElement("tr");
    if (quote.implied_vol === null) {
      row.className = "unavailable";
    }

    appendCell(row, quote.strike.toFixed(1), "num");
    appendCell(row, quote.option_type === "call" ? "Call" : "Put");
    appendCell(row, quote.bid.toFixed(2), "num");
    appendCell(row, quote.ask.toFixed(2), "num");
    appendCell(row, quote.mid.toFixed(2), "num");
    appendCell(row, quote.implied_vol === null
      ? "—"
      : `${(quote.implied_vol * 100).toFixed(2)}%`, "num");
    appendCell(row, quote.yahoo_iv === null
      ? "—"
      : `${(quote.yahoo_iv * 100).toFixed(2)}%`, "num");
    appendCell(row, quote.log_moneyness.toFixed(4), "num");

    row.title = quote.iv_note ?? "";
    body.append(row);
  }

  setStatus(chainNote,
    `${chain.expiry} · T = ${chain.days_to_expiry} дн. · `
    + `S = ${chain.spot.toFixed(2)}, форвард ${chain.forward.toFixed(2)} · `
    + `котировок ${chain.quoted}, волатильность найдена у ${chain.solved}`);
}


// --- график улыбки --------------------------------------------------
//
// Рисует plotly-basic (app/static/vendor/plotly-basic.min.js) — урезанная
// сборка без 3D и карт. Лежит файлом в проекте, а не тянется с CDN:
// страница должна открываться без интернета.

// Цвета повторяют те, что заданы в style.css для .chart. Plotly принимает
// только настоящие значения, переменные CSS ему передать нельзя.
const CHART_COLORS = {
  // Ключи call/put используются улыбкой напрямую по типу опциона;
  // primary — тот же слот 1 для графиков с единственным рядом.
  call: "#2a78d6",
  put: "#eb6834",
  primary: "#2a78d6",
  surface: "#ffffff",
  grid: "#e8eaed",
  text: "#1c1e21",
  muted: "#6b7280",
};

const CHART_FONT = '-apple-system, system-ui, "Segoe UI", sans-serif';


function renderSmile(chain) {
  const figure = document.getElementById("smile");
  const plot = document.getElementById("smile-plot");
  const points = chain.quotes.filter((q) => q.implied_vol !== null);

  if (points.length < 3) {
    figure.hidden = true;
    Plotly.purge(plot);
    return;
  }

  figure.hidden = false;
  document.getElementById("smile-caption").textContent =
    `Улыбка волатильности · ${chain.ticker} · экспирация ${chain.expiry}`;

  Plotly.newPlot(plot, smileTraces(points), smileLayout(points), {
    // Панель инструментов скрыта, но масштабирование протягиванием и
    // сброс двойным щелчком работают — на 130 точках это пригождается.
    displayModeBar: false,
    responsive: true,
  });
}


function smileTraces(points) {
  return ["call", "put"].map((side) => {
    const subset = points.filter((q) => q.option_type === side);

    return {
      type: "scatter",
      mode: "markers",
      name: side === "call" ? "Call" : "Put",
      x: subset.map((q) => q.log_moneyness),
      y: subset.map((q) => q.implied_vol * 100),
      customdata: subset.map((q) => [q.strike, q.mid]),
      marker: {
        size: 9,
        color: CHART_COLORS[side],
        // Кольцо цвета фона: около денег точки идут вплотную и без него
        // сливались бы в пятно.
        line: { width: 2, color: CHART_COLORS.surface },
      },
      // Значение ведёт, подпись вторична: ряд читатель уже знает по цвету
      // и легенде, ему нужно число.
      hovertemplate:
        "<b>%{y:.2f}%</b><br>страйк %{customdata[0]}<br>"
        + "цена %{customdata[1]:.2f}<extra></extra>",
    };
  });
}


function smileLayout(points) {
  const atm = points.reduce((best, q) =>
    Math.abs(q.log_moneyness) < Math.abs(best.log_moneyness) ? q : best
  );

  const axis = {
    gridcolor: CHART_COLORS.grid,
    gridwidth: 1,
    zeroline: false,
    linecolor: CHART_COLORS.grid,
    tickfont: { size: 11, color: CHART_COLORS.muted },
    titlefont: { size: 11, color: CHART_COLORS.muted },
  };

  return {
    height: 360,
    margin: { l: 56, r: 16, t: 12, b: 56 },
    paper_bgcolor: CHART_COLORS.surface,
    plot_bgcolor: CHART_COLORS.surface,
    font: { family: CHART_FONT, color: CHART_COLORS.text },
    hovermode: "closest",
    hoverlabel: {
      bgcolor: CHART_COLORS.surface,
      bordercolor: CHART_COLORS.grid,
      font: { family: CHART_FONT, size: 12, color: CHART_COLORS.text },
    },
    xaxis: {
      ...axis,
      title: { text: "ln(страйк / форвард) — ноль значит «на деньгах»" },
    },
    yaxis: { ...axis, title: { text: "волатильность" }, ticksuffix: "%" },
    legend: {
      orientation: "h",
      x: 0, y: 1.12,
      font: { size: 12, color: CHART_COLORS.muted },
    },
    shapes: [
      {
        // Вертикаль на нуле — граница «на деньгах». Сплошная волосяная:
        // пунктир читался бы как прогноз.
        type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1,
        line: { color: CHART_COLORS.muted, width: 1 }, opacity: 0.45,
      },
    ],
    annotations: [
      {
        // Единственная прямая подпись — уровень на деньгах. Числа у каждой
        // из 130 точек превратили бы график в кашу. Буквальный минимум
        // подписывать нельзя: у неликвидных страйков им регулярно
        // оказывается одиночная выпадающая точка, а не дно улыбки.
        x: atm.log_moneyness,
        y: atm.implied_vol * 100,
        text: `на деньгах ${(atm.implied_vol * 100).toFixed(1)}%`,
        showarrow: false,
        yshift: 18,
        font: { size: 11, color: CHART_COLORS.text },
      },
    ],
  };
}


// --- сходимость моделей ---------------------------------------------

async function loadConvergence() {
  if (!validate()) {
    setStatus(convergenceNote, "Сначала исправьте параметры выше", true);
    return;
  }

  const button = document.getElementById("load-convergence");
  button.disabled = true;
  setStatus(convergenceNote, "Считаем полсотни деревьев и десять симуляций…");

  try {
    const report = await postJSON("/api/price/convergence", {
      ...readParams(),
      model: selectedModel,
    });
    renderTreeConvergence(report);
    renderMonteCarloConvergence(report);

    const skipped = report.skipped_steps.length
      ? ` · пропущено шагов: ${report.skipped_steps.join(", ")}`
      : "";
    setStatus(convergenceNote,
      `Эталон — Блэк-Шоулз: ${report.reference.toFixed(6)} · `
      + `T = ${report.days_to_expiry} дн.${skipped}`);
  } catch (error) {
    document.getElementById("conv-tree").hidden = true;
    document.getElementById("conv-mc").hidden = true;
    setStatus(convergenceNote, error.message, true);
  } finally {
    button.disabled = false;
  }
}


/** Общая часть оформления обоих графиков сходимости. */
function convergenceLayout(reference, xTitle) {
  const axis = {
    gridcolor: CHART_COLORS.grid,
    gridwidth: 1,
    zeroline: false,
    linecolor: CHART_COLORS.grid,
    tickfont: { size: 11, color: CHART_COLORS.muted },
    titlefont: { size: 11, color: CHART_COLORS.muted },
  };

  return {
    height: 320,
    margin: { l: 64, r: 16, t: 12, b: 56 },
    paper_bgcolor: CHART_COLORS.surface,
    plot_bgcolor: CHART_COLORS.surface,
    font: { family: CHART_FONT, color: CHART_COLORS.text },
    hovermode: "closest",
    hoverlabel: {
      bgcolor: CHART_COLORS.surface,
      bordercolor: CHART_COLORS.grid,
      font: { family: CHART_FONT, size: 12, color: CHART_COLORS.text },
    },
    showlegend: false,
    // Ось логарифмическая: лестница идёт от 5 до двух тысяч, на линейной
    // шкале вся пила слиплась бы у левого края.
    xaxis: { ...axis, type: "log", title: { text: xTitle } },
    yaxis: { ...axis, title: { text: "цена опциона" } },
    shapes: [
      {
        // Точное значение — сплошная волосяная линия, а не ряд: это не
        // данные модели, а то, к чему она стремится.
        type: "line", yref: "y", y0: reference, y1: reference,
        xref: "paper", x0: 0, x1: 1,
        line: { color: CHART_COLORS.muted, width: 1 }, opacity: 0.6,
      },
    ],
    annotations: [
      {
        xref: "paper", x: 1, y: reference,
        text: `Блэк-Шоулз ${reference.toFixed(4)}`,
        showarrow: false, xanchor: "right", yshift: 10,
        font: { size: 11, color: CHART_COLORS.muted },
      },
    ],
  };
}


function renderTreeConvergence(report) {
  document.getElementById("conv-tree").hidden = false;

  const trace = {
    type: "scatter",
    mode: "lines+markers",
    x: report.tree.map((p) => p.steps),
    y: report.tree.map((p) => p.price),
    customdata: report.tree.map((p) => [p.error, p.elapsed_ms]),
    line: { width: 2, color: CHART_COLORS.primary },
    marker: { size: 6, color: CHART_COLORS.primary },
    hovertemplate:
      "<b>%{y:.4f}</b><br>шагов %{x}<br>отклонение %{customdata[0]:+.4f}"
      + "<br>%{customdata[1]:.1f} мс<extra></extra>",
  };

  Plotly.newPlot(
    document.getElementById("conv-tree-plot"),
    [trace],
    convergenceLayout(report.reference, "число шагов дерева"),
    { displayModeBar: false, responsive: true }
  );
}


function renderMonteCarloConvergence(report) {
  document.getElementById("conv-mc").hidden = false;

  const paths = report.monte_carlo.map((p) => p.paths);

  // Доверительный интервал — заливка между двумя границами. Рисуется
  // первой, чтобы линия цены легла поверх неё.
  const band = {
    type: "scatter",
    mode: "lines",
    x: [...paths, ...paths.slice().reverse()],
    y: [
      ...report.monte_carlo.map((p) => p.ci95_high),
      ...report.monte_carlo.map((p) => p.ci95_low).reverse(),
    ],
    fill: "toself",
    fillcolor: "rgba(42, 120, 214, 0.12)",
    line: { width: 0 },
    hoverinfo: "skip",
  };

  const line = {
    type: "scatter",
    mode: "lines+markers",
    x: paths,
    y: report.monte_carlo.map((p) => p.price),
    customdata: report.monte_carlo.map((p) => [p.error, p.std_error, p.elapsed_ms]),
    line: { width: 2, color: CHART_COLORS.primary },
    marker: { size: 8, color: CHART_COLORS.primary },
    hovertemplate:
      "<b>%{y:.4f}</b><br>путей %{x}<br>отклонение %{customdata[0]:+.4f}"
      + "<br>станд. ошибка %{customdata[1]:.4f}<br>%{customdata[2]:.0f} мс"
      + "<extra></extra>",
  };

  Plotly.newPlot(
    document.getElementById("conv-mc-plot"),
    [band, line],
    convergenceLayout(report.reference, "число путей"),
    { displayModeBar: false, responsive: true }
  );
}


// --- сверка с рынком ------------------------------------------------

/** Паритет колла и пута: проверка входных данных, а не моделей.
 *
 * Все остальные проверки в проекте внутренние — они отвечают на вопрос
 * «правильно ли посчитана формула». Эта отвечает на другой: «верны ли
 * числа, которые мы в формулу подставили». Волатильность в паритет не
 * входит, поэтому проверка не может выродиться в тавтологию.
 *
 * Запрос к GET /api/instruments/{ticker}/parity, математика — в
 * app/pricing/parity.py.
 */
async function loadParity() {
  const ticker = select.value;
  const expiry = document.getElementById("p-expiry").value;

  if (!ticker || !expiry) {
    return;
  }

  const button = document.getElementById("load-parity");
  button.disabled = true;
  setStatus(parityNote, `Сверяем цепочку на ${expiry}…`);

  try {
    const report = await getJSON(
      `/api/instruments/${ticker}/parity?expiry=${expiry}`
    );
    renderParity(report);
  } catch (error) {
    paritySummary.hidden = true;
    setStatus(parityNote, error.message, true);
  } finally {
    button.disabled = false;
  }
}


function renderParity(report) {
  const currency = snapshot?.currency ?? "USD";

  setText("pa-forward-ours", formatMoney(report.forward_ours, currency));
  setText("pa-forward-market", report.forward_market === null
    ? "нет данных"
    : `${formatMoney(report.forward_market, currency)} ± `
      + `${report.forward_tolerance.toFixed(2)}`);

  setText("pa-div-ours", formatPercent(report.div_yield_ours));
  setText("pa-div-market", formatPercent(report.div_yield_market));
  setText("pa-agreed", `${report.agreed} из ${report.pairs}`);

  const verdict = document.getElementById("parity-verdict");
  setStatus(verdict, report.verdict, report.agreed < report.pairs);

  renderParityRows(report.rows);

  paritySummary.hidden = false;
  setStatus(parityNote, parityFootnote(report));
}


function renderParityRows(rows) {
  const body = document.querySelector("#parity-table tbody");
  body.innerHTML = "";

  for (const row of rows) {
    const line = document.createElement("tr");
    if (!row.within_quotes) {
      line.className = "unavailable";
    }

    appendCell(line, row.strike.toFixed(1), "num");
    appendCell(line, row.call_mid.toFixed(2), "num");
    appendCell(line, row.put_mid.toFixed(2), "num");
    appendCell(line, row.observed.toFixed(2), "num");
    appendCell(line, row.expected.toFixed(2), "num");
    appendCell(line, formatGap(row.residual), "num");
    appendCell(line, row.tolerance.toFixed(2), "num");
    appendCell(line, row.implied_forward.toFixed(2), "num");
    appendCell(line, parityRowVerdict(row));

    line.title = row.arbitrage ?? "";
    body.append(line);
  }
}


function parityRowVerdict(row) {
  if (row.arbitrage) {
    return "вне границ";
  }
  return row.within_quotes ? "сходится" : "расходится";
}


/** Пояснение под таблицей: чего от этих чисел ждать, а чего нет. */
function parityFootnote(report) {
  const parts = [`T = ${report.days_to_expiry} дн.`];

  if (report.arbitrage > 0) {
    parts.push(
      `${report.arbitrage} страйков вне границ паритета — обычно это `
      + `разрыв в котировках или остаток досплитной серии`
    );
  }

  // Расхождение на дальних страйках закономерно и моделью не лечится:
  // на американские акции торгуются американские опционы, а у них
  // паритет — неравенство. Сильнее всего это видно на глубоко в деньгах.
  parts.push(
    "Расхождение обычно растёт к краям: там опцион глубоко в деньгах, "
    + "и в цене сидит премия за право исполнить его досрочно, которой "
    + "в европейском паритете нет"
  );

  return parts.join(" · ");
}


// --- вспомогательное ------------------------------------------------

/** Токен для эндпоинтов, меняющих состояние.
 *
 * Сервер подставляет его в мета-тег при отдаче страницы. Держать его в
 * этом файле нельзя: app.js раздаётся как статика и лежит в репозитории.
 */
function adminToken() {
  return document.querySelector('meta[name="admin-token"]')?.content ?? "";
}


async function postJSON(url, payload, { admin = false } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (admin) {
    headers["X-Admin-Token"] = adminToken();
  }

  const response = await fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(detailText(body) ?? `сервер ответил ${response.status}`);
  }

  return response.json();
}


/** detail бывает строкой, а бывает массивом ошибок валидации. */
function detailText(body) {
  const detail = body?.detail;

  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg ?? String(item)).join("; ");
  }
  return null;
}


async function getJSON(url) {
  const response = await fetch(url);

  if (!response.ok) {
    // FastAPI кладёт человекочитаемую причину в detail.
    const body = await response.json().catch(() => null);
    throw new Error(detailText(body) ?? `сервер ответил ${response.status}`);
  }

  return response.json();
}


function formatMoney(value, currency) {
  return new Intl.NumberFormat("ru-RU", {
    style: "currency",
    currency: currency,
    maximumFractionDigits: 2,
  }).format(value);
}


function formatPercent(value) {
  if (value === null || value === undefined) {
    return "нет данных";
  }
  return `${(value * 100).toFixed(2)}%`;
}


/** Дата без времени: строку YYYY-MM-DD переставляем сами.
 *
 * Через Date её гонять нельзя — «2026-09-15» разбирается как полночь UTC,
 * и в браузере западнее Гринвича отсечка съедет на день назад.
 */
function formatDay(value) {
  const [year, month, day] = value.split("-");
  return `${day}.${month}.${year}`;
}


function formatDate(value) {
  return new Date(value).toLocaleDateString("ru-RU", {
    day: "2-digit", month: "2-digit", year: "numeric",
  });
}


function round(value, digits) {
  return Number(value.toFixed(digits));
}


function numberOf(id) {
  return Number(document.getElementById(id).value);
}


function setValue(id, value) {
  document.getElementById(id).value = value;
}


function checkedValue(name) {
  return paramsForm.querySelector(`input[name="${name}"]:checked`).value;
}


function setText(id, text) {
  document.getElementById(id).textContent = text;
}


function setStatus(element, text, isError = false) {
  element.textContent = text;
  element.classList.toggle("error", isError);
}


// --- события --------------------------------------------------------

select.addEventListener("change", () => {
  if (!select.value) {
    marketCard.hidden = true;
    paramsCard.hidden = true;
    calcCard.hidden = true;
    resultCard.hidden = true;
    compareCard.hidden = true;
    chainCard.hidden = true;
    parityCard.hidden = true;
    convergenceCard.hidden = true;
    snapshot = null;
    setStatus(instrumentStatus, "");
    return;
  }

  setStatus(instrumentStatus, `Выбрано: ${select.value}`);
  loadMarket(select.value);
});


// Смена экспирации меняет q: у нового срока другой набор отсечек внутри.
// Введённое вручную значение при этом затирается — обратно его вернёт
// «Сбросить», как и остальные поля.
document.getElementById("p-expiry").addEventListener("change", applyDividendYield);

paramsForm.addEventListener("input", refreshHints);
paramsForm.addEventListener("change", refreshHints);

// Форма ничего не отправляет: Enter не должен перезагружать страницу.
paramsForm.addEventListener("submit", (event) => event.preventDefault());

document.getElementById("reset-params").addEventListener("click", () => {
  if (snapshot) {
    fillParams(snapshot);
  }
});

for (const button of document.querySelectorAll(".btn[data-model]")) {
  button.addEventListener("click", () => selectModel(button.dataset.model));
}

document.getElementById("calculate").addEventListener("click", runModel);
document.getElementById("compare").addEventListener("click", runComparison);

// Быстрая подстановка волатильности из нужного окна истории.
for (const button of document.querySelectorAll("button[data-sigma]")) {
  button.addEventListener("click", () => {
    const value = snapshot?.[button.dataset.sigma];

    if (value === null || value === undefined) {
      setStatus(paramsStatus, "Для этого окна не хватило истории", true);
      return;
    }

    setValue("p-sigma", round(value * 100, 2));
    refreshHints();
  });
}

document.getElementById("load-chain").addEventListener("click", loadChain);
document.getElementById("load-parity").addEventListener("click", loadParity);
document.getElementById("load-convergence").addEventListener("click", loadConvergence);

document.getElementById("sync-instruments").addEventListener("click", syncInstruments);

document.getElementById("refresh-market").addEventListener("click", () => {
  if (select.value) {
    loadMarket(select.value, true);
  }
});



loadInstruments();
