const gameSelect = document.getElementById("gameSelect");
const p1Type = document.getElementById("p1Type");
const p2Type = document.getElementById("p2Type");
const p1Bot = document.getElementById("p1Bot");
const p2Bot = document.getElementById("p2Bot");
const p1Algo = document.getElementById("p1Algo");
const p2Algo = document.getElementById("p2Algo");
const p1Checkpoint = document.getElementById("p1Checkpoint");
const p2Checkpoint = document.getElementById("p2Checkpoint");
const p1Sims = document.getElementById("p1Sims");
const p2Sims = document.getElementById("p2Sims");
const startBtn = document.getElementById("startBtn");
const autoBtn = document.getElementById("autoBtn");
const autoPlay = document.getElementById("autoPlay");
const boardEl = document.getElementById("board");
const actionPanel = document.getElementById("actionPanel");
const statusEl = document.getElementById("status");
const turnInfo = document.getElementById("turnInfo");
const resultInfo = document.getElementById("resultInfo");
const checkpointNote = document.getElementById("checkpointNote");

let state = null;
let checkpoints = [];
let supportedAlgos = [];

const BOT_OPTIONS_BY_GAME = {
  tictactoe: [
    { value: "random", label: "Random" },
    { value: "v0", label: "Rule Bot" },
    { value: "heuristic_perfect", label: "Heuristic Perfect" },
    { value: "alpha_beta_pruning", label: "Alpha-Beta" },
  ],
  pig: [
    { value: "random", label: "Random" },
    { value: "hold_at_20", label: "Hold at 20" },
  ],
  backgammon: [
    { value: "random", label: "Random" },
  ],
};

function optionEl(value, label, extra = {}) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  Object.entries(extra).forEach(([key, val]) => {
    opt.dataset[key] = val;
  });
  return opt;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function getPlayerSpec(typeSelect, botSelect, algoSelect, checkpointSelect, simsInput) {
  const type = typeSelect.value;
  if (type === "human") {
    return { player_type: "human" };
  }
  if (type === "bot") {
    return { player_type: "bot", bot_type: botSelect.value };
  }
  if (type === "model") {
    const algo = algoSelect.value;
    const path = checkpointSelect.value;
    const ckpt = checkpoints.find((c) => c.path === path);
    const sims = simsInput && simsInput.value ? parseInt(simsInput.value, 10) : null;
    if (!ckpt) {
      return { player_type: "model", checkpoint: null, algo: algo || null, num_simulations: sims };
    }
    return {
      player_type: "model",
      checkpoint: ckpt.path,
      algo: ckpt.algo || algo,
      num_simulations: sims,
    };
  }
  return { player_type: "human" };
}

function fillBotSelect(typeSelect, botSelect) {
  botSelect.innerHTML = "";
  if (typeSelect.value === "bot") {
    const game = gameSelect.value;
    const bots = BOT_OPTIONS_BY_GAME[game] || [];
    bots.forEach((bot) => botSelect.appendChild(optionEl(bot.value, bot.label)));
    botSelect.disabled = bots.length === 0;
  } else {
    botSelect.disabled = true;
  }
}

function fillAlgoSelect(typeSelect, algoSelect) {
  algoSelect.innerHTML = "";
  if (typeSelect.value === "model") {
    supportedAlgos.forEach((algo) => algoSelect.appendChild(optionEl(algo, algo)));
    algoSelect.disabled = supportedAlgos.length === 0;
  } else {
    algoSelect.disabled = true;
  }
}

function fillCheckpointSelect(typeSelect, algoSelect, checkpointSelect) {
  checkpointSelect.innerHTML = "";
  if (typeSelect.value === "model") {
    const algo = algoSelect.value;
    const filtered = checkpoints.filter((ckpt) => !algo || ckpt.algo === algo);
    filtered.forEach((ckpt) => {
      const label = ckpt.supported ? ckpt.path : `${ckpt.path} (unsupported)`;
      const opt = optionEl(ckpt.path, label);
      if (!ckpt.supported) {
        opt.disabled = true;
      }
      checkpointSelect.appendChild(opt);
    });
    checkpointSelect.disabled = filtered.length === 0;
  } else {
    checkpointSelect.disabled = true;
  }
}

function toggleSimInput(typeSelect, simsInput) {
  simsInput.disabled = typeSelect.value !== "model";
  if (typeSelect.value !== "model") {
    simsInput.value = "";
  }
}

async function loadGames() {
  const res = await fetch("/api/games");
  const data = await res.json();
  gameSelect.innerHTML = "";
  data.games.forEach((game) => {
    gameSelect.appendChild(optionEl(game.name, game.label));
  });
}

async function loadCheckpoints() {
  const game = gameSelect.value;
  const res = await fetch(`/api/checkpoints?game=${game}`);
  const data = await res.json();
  checkpoints = data.checkpoints || [];
  supportedAlgos = data.supported_algos || [];
  const supported = checkpoints.filter((ckpt) => ckpt.supported).length;
  if (checkpoints.length === 0) {
    checkpointNote.textContent = "No checkpoints found. Bots only.";
  } else {
    checkpointNote.textContent = `${checkpoints.length} checkpoints found (${supported} supported).`;
  }
  fillBotSelect(p1Type, p1Bot);
  fillBotSelect(p2Type, p2Bot);
  fillAlgoSelect(p1Type, p1Algo);
  fillAlgoSelect(p2Type, p2Algo);
  fillCheckpointSelect(p1Type, p1Algo, p1Checkpoint);
  fillCheckpointSelect(p2Type, p2Algo, p2Checkpoint);
}

function renderBoard() {
  boardEl.innerHTML = "";
  actionPanel.innerHTML = "";
  if (!state) {
    return;
  }
  if (state.game === "tictactoe") {
    boardEl.className = "board";
    const board = state.board;
    const legal = new Set(state.legal_actions || []);
    board.flat().forEach((value, idx) => {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.dataset.index = idx;
      let label = "";
      if (value === 1) {
        label = "X";
        cell.classList.add("x");
      } else if (value === 2) {
        label = "O";
        cell.classList.add("o");
      }
      cell.textContent = label;
      if (!state.done && legal.has(idx) && isHumanTurn()) {
        cell.classList.add("clickable");
        cell.addEventListener("click", () => handleMove(idx));
      } else {
        cell.disabled = true;
      }
      boardEl.appendChild(cell);
    });
    return;
  }

  boardEl.className = "board board-plain";
  if (state.game === "pig") {
    const stats = document.createElement("div");
    stats.className = "stats";
    stats.innerHTML = `
      <div><strong>Player 1</strong> score: ${state.scores[0]}</div>
      <div><strong>Player 2</strong> score: ${state.scores[1]}</div>
      <div><strong>Turn score</strong>: ${state.turn_score}</div>
      <div><strong>Legal actions</strong>: ${state.legal_actions.join(", ") || "none"}</div>
    `;
    boardEl.appendChild(stats);

    const canRoll = state.legal_actions.includes(0);
    const canHold = state.legal_actions.includes(1);
    if (!state.done && isHumanTurn()) {
      if (canRoll) {
        const rollBtn = document.createElement("button");
        rollBtn.textContent = "Roll";
        rollBtn.addEventListener("click", () => handleMove(0));
        actionPanel.appendChild(rollBtn);
      }
      if (canHold) {
        const holdBtn = document.createElement("button");
        holdBtn.textContent = "Hold";
        holdBtn.className = "ghost";
        holdBtn.addEventListener("click", () => handleMove(1));
        actionPanel.appendChild(holdBtn);
      }
    }
    return;
  }

  if (state.game === "backgammon") {
    const dice = state.dice || {};
    const header = document.createElement("div");
    header.className = "stats";
    header.innerHTML = `
      <div><strong>Dice</strong>: ${dice.remaining ? dice.remaining.join(", ") : "-"}</div>
      <div><strong>Slots</strong>: ${dice.slots ? dice.slots.join(", ") : "-"}</div>
      <div><strong>Player 1 bar</strong>: ${state.board.p1_bar}</div>
      <div><strong>Player 2 bar</strong>: ${state.board.p2_bar}</div>
      <div><strong>Player 1 off</strong>: ${state.board.p1_off}</div>
      <div><strong>Player 2 off</strong>: ${state.board.p2_off}</div>
    `;
    boardEl.appendChild(header);

    const bg = document.createElement("div");
    bg.className = "bg-board";
    const myPoints = state.board.p1_points || [];
    const oppPoints = state.board.p2_points || [];
    const topRow = document.createElement("div");
    topRow.className = "bg-row";
    const bottomRow = document.createElement("div");
    bottomRow.className = "bg-row";

    function makePoint(idx) {
      const cell = document.createElement("div");
      cell.className = "bg-point";
      const label = document.createElement("div");
      label.className = "bg-label";
      label.textContent = `${idx + 1}`;
      cell.appendChild(label);

      const stack = document.createElement("div");
      stack.className = "bg-stack";
      const mine = myPoints[idx] || 0;
      const opp = oppPoints[idx] || 0;
      for (let i = 0; i < Math.min(mine, 5); i += 1) {
        const token = document.createElement("span");
        token.className = "bg-checker mine";
        stack.appendChild(token);
      }
      if (mine > 5) {
        const counter = document.createElement("div");
        counter.className = "bg-counter mine";
        counter.textContent = `+${mine - 5}`;
        stack.appendChild(counter);
      }
      for (let i = 0; i < Math.min(opp, 5); i += 1) {
        const token = document.createElement("span");
        token.className = "bg-checker opp";
        stack.appendChild(token);
      }
      if (opp > 5) {
        const counter = document.createElement("div");
        counter.className = "bg-counter opp";
        counter.textContent = `+${opp - 5}`;
        stack.appendChild(counter);
      }
      cell.appendChild(stack);
      return cell;
    }

    for (let idx = 23; idx >= 12; idx -= 1) {
      topRow.appendChild(makePoint(idx));
    }
    for (let idx = 0; idx <= 11; idx += 1) {
      bottomRow.appendChild(makePoint(idx));
    }
    bg.appendChild(topRow);
    bg.appendChild(bottomRow);
    boardEl.appendChild(bg);

    if (!state.done && isHumanTurn()) {
      if (state.legal_action_details && state.legal_action_details.length > 0) {
        const select = document.createElement("select");
        state.legal_action_details.forEach((entry) => {
          const opt = document.createElement("option");
          opt.value = entry.action;
          opt.textContent = entry.label;
          select.appendChild(opt);
        });
        actionPanel.appendChild(select);
        const playBtn = document.createElement("button");
        playBtn.textContent = "Play Action";
        playBtn.addEventListener("click", () => handleMove(parseInt(select.value, 10)));
        actionPanel.appendChild(playBtn);
      } else {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = "No legal actions. Use Advance AI.";
        actionPanel.appendChild(pill);
      }
    }
  }
}

function isHumanTurn() {
  if (!state) return false;
  const spec = state.players[state.current_player];
  return spec && spec.player_type === "human";
}

function renderStatus() {
  if (!state) {
    setStatus("No active session");
    turnInfo.textContent = "Turn: --";
    resultInfo.textContent = "Result: --";
    return;
  }
  let current = `Player ${state.current_player}`;
  if (state.game === "tictactoe") {
    current = state.current_player === 1 ? "X" : "O";
  }
  const winner = state.winner;
  if (state.done) {
    if (winner === -1) {
      resultInfo.textContent = "Result: Draw";
    } else {
      if (state.game === "tictactoe") {
        const symbol = winner === 1 ? "X" : "O";
        resultInfo.textContent = `Result: ${symbol} wins`;
      } else {
        const player = winner === 0 ? 1 : winner;
        resultInfo.textContent = `Result: Player ${player} wins`;
      }
    }
  } else {
    resultInfo.textContent = "Result: In progress";
  }
  turnInfo.textContent = `Turn: ${current}`;
  const mode = isHumanTurn() ? "waiting for you" : "AI thinking";
  setStatus(state.done ? "Game over" : `Session active (${mode})`);
}

async function startSession() {
  const payload = {
    game: gameSelect.value,
    auto_play: autoPlay.checked,
    players: {
      1: getPlayerSpec(p1Type, p1Bot, p1Algo, p1Checkpoint, p1Sims),
      2: getPlayerSpec(p2Type, p2Bot, p2Algo, p2Checkpoint, p2Sims),
    },
  };
  const res = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  state = data.state;
  renderBoard();
  renderStatus();
}

async function handleMove(action) {
  const res = await fetch("/api/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  state = data.state;
  renderBoard();
  renderStatus();
}

async function advanceAI() {
  const res = await fetch("/api/auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    return;
  }
  state = data.state;
  renderBoard();
  renderStatus();
}

p1Type.addEventListener("change", () => {
  fillBotSelect(p1Type, p1Bot);
  fillAlgoSelect(p1Type, p1Algo);
  fillCheckpointSelect(p1Type, p1Algo, p1Checkpoint);
  toggleSimInput(p1Type, p1Sims);
});
p2Type.addEventListener("change", () => {
  fillBotSelect(p2Type, p2Bot);
  fillAlgoSelect(p2Type, p2Algo);
  fillCheckpointSelect(p2Type, p2Algo, p2Checkpoint);
  toggleSimInput(p2Type, p2Sims);
});
p1Algo.addEventListener("change", () => fillCheckpointSelect(p1Type, p1Algo, p1Checkpoint));
p2Algo.addEventListener("change", () => fillCheckpointSelect(p2Type, p2Algo, p2Checkpoint));
startBtn.addEventListener("click", startSession);
autoBtn.addEventListener("click", advanceAI);

gameSelect.addEventListener("change", async () => {
  await loadCheckpoints();
});

(async function init() {
  await loadGames();
  await loadCheckpoints();
  fillBotSelect(p1Type, p1Bot);
  fillBotSelect(p2Type, p2Bot);
  fillAlgoSelect(p1Type, p1Algo);
  fillAlgoSelect(p2Type, p2Algo);
  fillCheckpointSelect(p1Type, p1Algo, p1Checkpoint);
  fillCheckpointSelect(p2Type, p2Algo, p2Checkpoint);
  toggleSimInput(p1Type, p1Sims);
  toggleSimInput(p2Type, p2Sims);
  renderBoard();
  renderStatus();
})();
