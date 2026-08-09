/**
 * Model integration boundary.
 * Replace the body of `runModel()` with your real estimator.
 */
export const MODEL_PROFILES = [
  { id: "logistic-regression", name: "Logistic Regression" },
  { id: "random-forest", name: "Random Forest" },
  { id: "xgboost", name: "XGBoost" },
  { id: "dnn", name: "Deep Neural Network" },
];

// Memoizes predictions so re-renders don't re-run the model.
const predictionCache = new Map();

export function clearPredictionCache() {
  predictionCache.clear();
}

/** Actual model call. Returns P(playerA wins). */
function runModel(playerA, playerB, context) {
  const modelType = context.estimator ?? MODEL_PROFILES[0].id;
  
  switch (modelType) {
    case "logistic-regression":
      return 0

    case "random_forest":
      return 0

    case "xgboost":
      return 0
      
    case "dnn":
      return 0
      
    default:
      return 0
  }
}

export function predictMatch(playerA, playerB, context = {}) {
  if (!playerA || !playerB) return 0.5;

  const estimator = context.estimator ?? MODEL_PROFILES[0].id;
  // Order-independent key so (A,B) and (B,A) share one entry.
  const [lo, hi] = playerA.id <= playerB.id ? [playerA, playerB] : [playerB, playerA];
  const key = `${estimator}|${lo.id}|${hi.id}|${context.surface ?? "any"}`;

  if (!predictionCache.has(key)) {
    predictionCache.set(key, runModel(lo, hi, context));
  }
  const pLo = predictionCache.get(key);
  return playerA.id === lo.id ? pLo : 1 - pLo;
}

export function describeModel(modelId = MODEL_PROFILES[0].id) {
  return MODEL_PROFILES.find((profile) => profile.id === modelId)?.name || MODEL_PROFILES[0].name;
}