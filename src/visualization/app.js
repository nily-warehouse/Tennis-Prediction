import { describeModel, MODEL_PROFILES, predictMatch, clearPredictionCache } from "./model-adapter.js";
import { PLAYER_POOL } from "./players.js";

const state = {
  players: [],
  rounds: [],
  played: [],
  estimator: MODEL_PROFILES[0].id,
};

const $ = (selector) => document.querySelector(selector);
const formatPct = (value) => `${Math.round(value * 100)}%`;
const initials = (name) => name.split(" ").map((part) => part[0]).join("").slice(0, 2);
const predictionContext = (context = {}) => ({ ...context, estimator: state.estimator });

function nextPowerOfTwo(value) {
  return 2 ** Math.ceil(Math.log2(value));
}

function createLeague(count) {
  clearPredictionCache();

  state.players = PLAYER_POOL.slice(0, count).map((player, index) => ({
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

function simulateFixture(fixture) {
  const probability = predictMatch(
    fixture.a,
    fixture.b,
    predictionContext({ surface: "hard", tournament: state.players }),
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

function playMatch() {
  const fixture = nextMatch();
  if (!fixture) return;
  simulateFixture(fixture);
  render();
}

function playAll() {
  let champion = null;
  let fixture = nextMatch();
  while (fixture) {
    champion = simulateFixture(fixture).winner;
    fixture = nextMatch();
  }
  if (!champion) return;
  render();
}

function standings() {
  return [...state.players].sort(
    (a, b) => b.points - a.points || b.wins - a.wins || b.generalElo - a.generalElo,
  );
}

function render() {
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
  $("#model-name").textContent = describeModel(state.estimator);
  $("#play-one").disabled = !next;
  $("#play-all").disabled = !next;

  const probability = next ? predictMatch(next.a, next.b, predictionContext({ surface: "hard" })) : 0.5;
  $("#prob-a").style.width = `${probability * 100}%`;
  $("#prob-b").style.width = `${(1 - probability) * 100}%`;
  $("#prob-a-label").textContent = next ? formatPct(probability) : "—";
  $("#prob-b-label").textContent = next ? formatPct(1 - probability) : "—";
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
          ? predictMatch(match.a, match.b, predictionContext({ surface: "hard" }))
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
