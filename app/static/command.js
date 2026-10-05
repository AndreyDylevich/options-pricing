/* Командная строка.
 *
 * Загружается после app.js и вызывает его функции напрямую: сборки в
 * проекте нет, оба файла живут в одной области видимости. Своих запросов к
 * API здесь нет ни одного — команда лишь заполняет те же поля формы и
 * нажимает те же кнопки, что и мышь. Поэтому проверки ввода, ограничения
 * частоты и обработка ошибок достаются ей бесплатно, а не дублируются.
 *
 * Аргументы необязательны: команда без них работает на том, что уже в
 * форме. `price` пересчитает как есть, `price 345` поменяет только страйк.
 */

const consoleLog = document.getElementById("console-log");
const consoleInput = document.getElementById("console-input");

// История команд: в терминале её ждут на стрелках вверх и вниз.
const commandHistory = [];
let historyCursor = 0;

// Сколько строк держать в журнале. Больше незачем: журнал под рукой для
// последних действий, а не архив.
const LOG_LIMIT = 60;


function echo(text, className = "") {
  const line = document.createElement("span");
  line.textContent = `${text}\n`;
  if (className) {
    line.className = className;
  }
  consoleLog.append(line);

  while (consoleLog.childElementCount > LOG_LIMIT) {
    consoleLog.firstElementChild.remove();
  }

  // Прокручиваем к последней строке: журнал невысокий, и без этого свежий
  // вывод оказывался бы за краем.
  consoleLog.scrollTop = consoleLog.scrollHeight;
}


/** Разбор строки: имя команды, позиционные слова, ключи вида --name value. */
function parse(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const positional = [];
  const options = {};

  for (let index = 1; index < words.length; index += 1) {
    const word = words[index];
    if (word.startsWith("--")) {
      options[word.slice(2)] = words[index + 1] ?? "";
      index += 1;
    } else {
      positional.push(word);
    }
  }

  return { name: (words[0] ?? "").toLowerCase(), positional, options };
}


/** Выбрать бумагу: то же, что выбрать её в списке. */
async function cmdUse(args) {
  const ticker = (args.positional[0] ?? "").toUpperCase();
  if (!ticker) {
    throw new Error("нужен тикер: use AAPL");
  }

  const known = [...select.options].some((option) => option.value === ticker);
  if (!known) {
    throw new Error(`${ticker} нет в списке — доступные смотрите в «Базовый актив»`);
  }

  select.value = ticker;
  // Событие, а не прямой вызов: обработчик выбора делает не только
  // загрузку данных, и повторять его содержимое здесь значило бы
  // разойтись с ним при первой же правке.
  select.dispatchEvent(new Event("change"));
  echo(`выбрано ${ticker}, тянем рыночные данные`);
}


/** Подставить в форму то, что указано позиционно и ключами. */
function applyParams(args) {
  const [strike, expiry, type] = args.positional;
  const changed = [];

  if (strike !== undefined) {
    if (!Number.isFinite(Number(strike))) {
      throw new Error(`страйк «${strike}» не число`);
    }
    setValue("p-strike", Number(strike));
    changed.push(`страйк ${strike}`);
  }

  if (expiry !== undefined) {
    const field = document.getElementById("p-expiry");
    const listed = [...field.options].some((option) => option.value === expiry);
    if (!listed) {
      throw new Error(`экспирации ${expiry} нет у этой бумаги`);
    }
    field.value = expiry;
    // Смена экспирации меняет q: тот же пересчёт, что и при выборе мышью.
    field.dispatchEvent(new Event("change"));
    changed.push(`экспирация ${expiry}`);
  }

  if (type !== undefined) {
    const wanted = type.toLowerCase();
    if (wanted !== "call" && wanted !== "put") {
      throw new Error(`тип «${type}»: ожидается call или put`);
    }
    paramsForm.querySelector(`input[name="option_type"][value="${wanted}"]`)
      .checked = true;
    changed.push(wanted);
  }

  if (args.options.sigma !== undefined) {
    setValue("p-sigma", Number(args.options.sigma));
    changed.push(`σ ${args.options.sigma}%`);
  }

  if (args.options.style !== undefined) {
    const style = args.options.style.toLowerCase();
    if (style !== "european" && style !== "american") {
      throw new Error(`стиль «${style}»: ожидается european или american`);
    }
    paramsForm.querySelector(`input[name="exercise"][value="${style}"]`)
      .checked = true;
    syncModelsWithExercise();
    changed.push(style);
  }

  if (changed.length > 0) {
    refreshHints();
  }

  return changed;
}


const MODEL_ALIASES = {
  bs: "black_scholes",
  "black-scholes": "black_scholes",
  black_scholes: "black_scholes",
  tree: "binomial",
  binomial: "binomial",
  mc: "monte_carlo",
  "monte-carlo": "monte_carlo",
  monte_carlo: "monte_carlo",
};


async function cmdPrice(args) {
  if (!select.value) {
    throw new Error("сначала выберите бумагу: use AAPL");
  }

  const changed = applyParams(args);

  if (args.options.model !== undefined) {
    const model = MODEL_ALIASES[args.options.model.toLowerCase()];
    if (!model) {
      throw new Error(
        `модель «${args.options.model}»: ожидается bs, tree или mc`
      );
    }
    selectModel(model);
    changed.push(MODEL_NAMES[model]);
  }

  if (changed.length > 0) {
    echo(`подставлено: ${changed.join(", ")}`);
  }

  await runModel();

  const price = document.getElementById("r-price").textContent;
  const market = document.getElementById("market-block").hidden
    ? null
    : document.getElementById("r-market").textContent;

  echo(market === null
    ? `цена ${price}`
    : `цена ${price}   рынок ${market}`);
}


async function cmdCompare(args) {
  if (!select.value) {
    throw new Error("сначала выберите бумагу: use AAPL");
  }
  applyParams(args);
  await runComparison();
  echo("сравнение готово, закладка «Сравнение»");
}


/** Команды, которые просто нажимают кнопку и открывают свою закладку. */
const PANEL_COMMANDS = {
  vol: { run: loadVolatility, pane: null, done: "таблица волатильности готова" },
  smile: { run: loadChain, pane: "chain-card", done: "улыбка построена" },
  greeks: { run: loadProfiles, pane: "profiles-card", done: "профили построены" },
  conv: { run: loadConvergence, pane: "convergence-card", done: "сходимость построена" },
  parity: { run: loadParity, pane: "parity-card", done: "сверка выполнена" },
};


async function cmdPanel(name, args) {
  if (!select.value) {
    throw new Error("сначала выберите бумагу: use AAPL");
  }

  const command = PANEL_COMMANDS[name];
  applyParams(args);

  // Закладку открываем до запуска, а не после: Plotly, нарисованный в
  // скрытом элементе, получает нулевую ширину и схлопывается.
  if (command.pane !== null) {
    showTab(command.pane);
  }

  await command.run();
  echo(command.done);
}


const HELP = [
  "use AAPL                       выбрать бумагу",
  "price [K] [дата] [call|put]    посчитать выбранной моделью",
  "   --model bs|tree|mc  --sigma 24.5  --style european|american",
  "compare [K] [дата] [call|put]  все три модели сразу",
  "vol                            таблица: историческая, GARCH, EGARCH",
  "smile                          цепочка опционов и улыбка",
  "greeks                         профили дельты, гаммы, теты",
  "conv                           сходимость дерева и Монте-Карло",
  "parity                         сверка наших данных с рынком",
  "clear                          очистить журнал",
  "",
  "Аргументы необязательны: без них команда берёт то, что в форме.",
];


async function execute(text) {
  const args = parse(text);

  if (args.name === "help" || args.name === "?") {
    HELP.forEach((line) => echo(line));
    return;
  }
  if (args.name === "clear") {
    consoleLog.innerHTML = "";
    return;
  }
  if (args.name === "use") {
    await cmdUse(args);
    return;
  }
  if (args.name === "price") {
    await cmdPrice(args);
    return;
  }
  if (args.name === "compare") {
    await cmdCompare(args);
    return;
  }
  if (args.name in PANEL_COMMANDS) {
    await cmdPanel(args.name, args);
    return;
  }

  throw new Error(`неизвестная команда «${args.name}» — наберите help`);
}


consoleInput.addEventListener("keydown", async (event) => {
  if (event.key === "ArrowUp") {
    event.preventDefault();
    historyCursor = Math.max(0, historyCursor - 1);
    consoleInput.value = commandHistory[historyCursor] ?? "";
    return;
  }

  if (event.key === "ArrowDown") {
    event.preventDefault();
    historyCursor = Math.min(commandHistory.length, historyCursor + 1);
    consoleInput.value = commandHistory[historyCursor] ?? "";
    return;
  }

  if (event.key !== "Enter") {
    return;
  }

  const text = consoleInput.value.trim();
  if (!text) {
    return;
  }

  commandHistory.push(text);
  historyCursor = commandHistory.length;
  consoleInput.value = "";
  consoleInput.disabled = true;
  echo(`> ${text}`, "echo");

  try {
    await execute(text);
  } catch (error) {
    echo(error.message, "fail");
  } finally {
    consoleInput.disabled = false;
    consoleInput.focus();
  }
});
