import { describeModel, MODEL_PROFILES } from "./model-adapter.js";
import { PLAYER_POOL } from "./players.js";

const state = {
  players: [],
  rounds: [],
  played: [],
  estimator: MODEL_PROFILES[0].id,
  loading: false,
  error: null,
};

const predictionCache = new Map();
const pendingPredictions = new Map();
const modelNames = {
  "logistic-regression": "logistic_regression",
  "random-forest": "random_forest",
  xgboost: "xgboost",
  dnn: "dnn",
};

const $ = (selector) => document.querySelector(selector);
const formatPct = (value) => `${Math.round(value * 100)}%`;
const initials = (name) => name.split(" ").map((part) => part[0]).join("").slice(0, 2);
const predictionContext = (context = {}) => ({ ...context, estimator: state.estimator });
const numericValue = (value, fallback = 0) => (
  typeof value === "number" && Number.isFinite(value) ? value : fallback
);

function playerData(player) {
  const generalElo = numericValue(player.generalElo);
  const surfaceElo = numericValue(player.surfaceElo, generalElo);
  return [
    player.name,
    generalElo,
    surfaceElo,
    numericValue(player.pts),
    numericValue(player.rank),
    numericValue(player.matches),
    numericValue(player.surfaceMatches),
    numericValue(player.effectiveElo, generalElo),
    numericValue(player.spec, surfaceElo - generalElo),
  ];
}

function predictionKey(playerA, playerB, context) {
  const [first, second] = playerA.id <= playerB.id ? [playerA, playerB] : [playerB, playerA];
  return `${context.estimator}|${first.id}|${second.id}|${context.surface ?? "any"}`;
}

async function requestPrediction(playerA, playerB, context) {
  const payload = {
    player1: playerData(playerA),
    player2: playerData(playerB),
    meta_data: [context.surface === "hard" ? "Hard" : context.surface, context.bestOf ?? 3],
    model: modelNames[context.estimator] ?? context.estimator,
  };
  const response = await fetch("/api/estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(result.error || `Estimator request failed (${response.status})`);
  }

  const probability = Number(result.probability);
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new Error("Estimator returned an invalid probability");
  }
  return probability;
}

async function predictMatch(playerA, playerB, context = {}) {
  if (!playerA || !playerB) return 0.5;

  const fullContext = predictionContext(context);
  const [first, second] = playerA.id <= playerB.id ? [playerA, playerB] : [playerB, playerA];
  const key = predictionKey(first, second, fullContext);

  if (!predictionCache.has(key)) {
    if (!pendingPredictions.has(key)) {
      pendingPredictions.set(
        key,
        requestPrediction(first, second, fullContext)
          .then((probability) => {
            predictionCache.set(key, probability);
            return probability;
          })
          .finally(() => pendingPredictions.delete(key)),
      );
    }
    await pendingPredictions.get(key);
  }

  const firstProbability = predictionCache.get(key);
  return playerA.id === first.id ? firstProbability : 1 - firstProbability;
}

function cachedPrediction(playerA, playerB, context = {}) {
  if (!playerA || !playerB) return null;
  const fullContext = predictionContext(context);
  const [first] = playerA.id <= playerB.id ? [playerA, playerB] : [playerB, playerA];
  const probability = predictionCache.get(predictionKey(playerA, playerB, fullContext));
  if (probability === undefined) return null;
  return playerA.id === first.id ? probability : 1 - probability;
}

function updateEstimatorStatus() {
  const modelName = describeModel(state.estimator);
  $("#model-name").textContent = state.error
    ? `Error: ${state.error}`
    : state.loading ? `Loading ${modelName}...` : modelName;
  $("#model-name").title = state.error || "";
  $("#league-size").disabled = state.loading;
  $("#estimator-model").disabled = state.loading;
  $("#new-league").disabled = state.loading;
  $("#play-one").disabled = state.loading || Boolean(state.error) || !nextMatch();
  $("#play-all").disabled = state.loading || Boolean(state.error) || !nextMatch();
}

function nextPowerOfTwo(value) {
  return 2 ** Math.ceil(Math.log2(value));
}

function shuffledPlayerPool() {
  const players = PLAYER_POOL.slice();
  for (let index = players.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [players[index], players[swapIndex]] = [players[swapIndex], players[index]];
  }
  return players;
}

function createLeague(count) {
  predictionCache.clear();
  pendingPredictions.clear();
  state.error = null;

  state.players = shuffledPlayerPool().slice(0, count).map((player, index) => ({
    ...player, id: index, played: 0, wins: 0, losses: 0, points: 0,
  }));
  state.played = [];

  const bracketSize = nextPowerOfTwo(count);
  state.rounds = [[]];
  for (let i = 0; i < bracketSize; i += 2) {
    state.rounds[0].push({
      round: 0, index: i / 2, a: state.players[i] || null, b: state.players[i + 1] || null,
      winner: null, played: false, resolved: false,
    });
  }
  for (let round = 1; round < Math.log2(bracketSize); round += 1) {
    state.rounds.push(Array.from({ length: state.rounds[round - 1].length / 2 }, (_, index) => ({
      round, index, a: null, b: null, winner: null, played: false, resolved: false,
      sourceA: state.rounds[round - 1][index * 2],
      sourceB: state.rounds[round - 1][index * 2 + 1],
    })));
  }

  resolveByes();
  render();
}

function matchIsReady(match) {
  return Boolean(match.winner || match.played || match.resolved);
}

function resolveByes() {
  let changed = true;
  while (changed) {
    changed = false;
    state.rounds.forEach((round) => round.forEach((match) => {
      if (match.round > 0) {
        match.a = match.sourceA.winner || null;
        match.b = match.sourceB.winner || null;
      }
      const sourceMatchesComplete = match.round === 0
        || (matchIsReady(match.sourceA) && matchIsReady(match.sourceB));

      if (!match.a && !match.b && !match.winner && !match.played && !match.resolved
        && sourceMatchesComplete) {
        match.resolved = true;
        changed = true;
      }
      if (!match.winner && !match.played && sourceMatchesComplete
        && ((match.a && !match.b) || (!match.a && match.b))) {
        match.winner = match.a || match.b;
        match.resolved = true;
        changed = true;
      }
    }));
  }
}

function nextMatch() {
  resolveByes();
  for (const round of state.rounds) {
    const match = round.find((candidate) => !candidate.winner && candidate.a && candidate.b);
    if (match) return match;
  }
  return null;
}

function bracketSlot(match, side) {
  if (match.round === 0) return (side === "a" ? match.a : match.b) || "";
  const source = side === "a" ? match.sourceA : match.sourceB;
  if (source.winner) return source.winner;
  return source.resolved ? "" : "Waiting";
}

async function simulateFixture(fixture) {
  const probability = await predictMatch(
    fixture.a,
    fixture.b,
    { surface: "hard", tournament: state.players },
  );
  const homeWins = Math.random() < probability;
  const winner = homeWins ? fixture.a : fixture.b;
  const loser = homeWins ? fixture.b : fixture.a;

  winner.played += 1;
  winner.wins += 1;
  winner.points += 2;
  loser.played += 1;
  loser.losses += 1;

  fixture.winner = winner;
  fixture.played = true;
  fixture.resolved = true;

  state.played.unshift({ ...fixture, winner, loser, probability, round: fixture.round + 1 });
  resolveByes();
  return { winner, loser, probability, homeWins };
}

async function playMatch() {
  const fixture = nextMatch();
  if (!fixture) return;
  state.loading = true;
  state.error = null;
  updateEstimatorStatus();
  try {
    await simulateFixture(fixture);
    await render();
  } catch (error) {
    state.loading = false;
    state.error = error.message || "Unable to contact the estimator";
    renderView();
  }
}

async function playAll() {
  let champion = null;
  let fixture = nextMatch();
  if (!fixture) return;
  state.loading = true;
  state.error = null;
  updateEstimatorStatus();
  try {
    while (fixture) {
      champion = (await simulateFixture(fixture)).winner;
      fixture = nextMatch();
    }
    if (!champion) return;
    await render();
  } catch (error) {
    state.loading = false;
    state.error = error.message || "Unable to contact the estimator";
    renderView();
  }
}

function standings() {
  return [...state.players].sort(
    (a, b) => b.points - a.points || b.wins - a.wins || b.generalElo - a.generalElo,
  );
}

async function render() {
  resolveByes();
  state.loading = true;
  state.error = null;
  updateEstimatorStatus();

  try {
    const predictions = [];
    state.rounds.forEach((round) => round.forEach((match) => {
      if (match.a && match.b) {
        predictions.push(predictMatch(match.a, match.b, { surface: "hard" }));
      }
    }));
    await Promise.all(predictions);
  } catch (error) {
    state.error = error.message || "Unable to contact the estimator";
  }

  state.loading = false;
  renderView();
}

function renderView() {
  resolveByes();

  $("#standings-body").innerHTML = standings().map((player, index) => `
    <tr><td class="rank">${String(index + 1).padStart(2, "0")}</td>
    <td><span class="avatar">${initials(player.name)}</span><span class="player-name">${player.name}</span></td>
    <td>${player.played}</td><td class="win">${player.wins}</td><td>${player.losses}</td><td class="points">${player.points}</td></tr>`).join("");

  const next = nextMatch();
  $("#next-match").innerHTML = next ? `${next.a.name} <span>vs</span> ${next.b.name}` : "Season complete";

  const totalMatches = Math.max(0, state.players.length - 1);
  $("#fixture-count").textContent = `${state.played.length} / ${totalMatches} matches played`;
  $("#progress").style.width = `${totalMatches ? (state.played.length / totalMatches) * 100 : 0}%`;
  updateEstimatorStatus();

  const probability = next ? cachedPrediction(next.a, next.b, { surface: "hard" }) : null;
  $("#prob-a").style.width = `${(probability ?? 0.5) * 100}%`;
  $("#prob-b").style.width = `${(1 - (probability ?? 0.5)) * 100}%`;
  $("#prob-a-label").textContent = probability !== null ? formatPct(probability) : "—";
  $("#prob-b-label").textContent = probability !== null ? formatPct(1 - probability) : "—";
  $("#player-a").textContent = next ? next.a.name : "—";
  $("#player-b").textContent = next ? next.b.name : "—";

  $("#bracket").innerHTML = state.rounds.map((round, index) => {
    const label = index === state.rounds.length - 1 ? "FINAL"
      : index === state.rounds.length - 2 ? "SEMIFINALS"
      : index === 0 ? `ROUND OF ${state.rounds[0].length * 2}`
      : `ROUND ${index + 1}`;
    return `
    <div class="bracket-round round-${index + 1}">
      <div class="round-label">${label}</div>
      <div class="round-matches">${round.map((match) => {
        if (match.resolved && !match.a && !match.b) return "";
        const a = bracketSlot(match, "a");
        const b = bracketSlot(match, "b");
        const winnerA = Boolean(match.winner && match.a && match.winner.id === match.a.id);
        const winnerB = Boolean(match.winner && match.b && match.winner.id === match.b.id);
        const edge = match.a && match.b
          ? cachedPrediction(match.a, match.b, { surface: "hard" })
          : null;
        return `<div class="bracket-match ${match.winner ? "is-complete" : ""} ${next === match ? "is-next" : ""}">
          <div class="bracket-player ${winnerA ? "is-winner" : ""}"><span>${typeof a === "string" ? a : a.name}</span>${winnerA ? "✓" : ""}</div>
          <div class="bracket-player ${winnerB ? "is-winner" : ""}"><span>${typeof b === "string" ? b : b.name}</span>${winnerB ? "✓" : ""}</div>
          ${edge !== null ? `<small>${formatPct(edge)} model edge</small>` : ""}
        </div>`;
      }).join("")}</div>
    </div>`;
  }).join("");

  $("#match-feed").innerHTML = state.played.slice(0, 7).map((match) => `
    <li><span class="feed-round">R${match.round}</span><span><strong>${match.winner.name}</strong> beat ${match.loser.name}</span><em>${formatPct(match.winner === match.a ? match.probability : 1 - match.probability)}</em></li>`).join("")
    || '<li class="empty">No matches played yet</li>';
}

$("#league-size").addEventListener("change", (event) => createLeague(Number(event.target.value)));
$("#new-league").addEventListener("click", () => createLeague(Number($("#league-size").value)));
$("#estimator-model").innerHTML = MODEL_PROFILES.map(({ id, name }, index) => `<option value="${id}"${index === 0 ? " selected" : ""}>${name}</option>`).join("");
$("#estimator-model").addEventListener("change", (event) => {
  state.estimator = event.target.value;
  createLeague(Number($("#league-size").value));
});
$("#play-one").addEventListener("click", playMatch);
$("#play-all").addEventListener("click", playAll);

createLeague(Number($("#league-size").value));
