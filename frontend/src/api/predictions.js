import client from "./client";

export const getPrediction = (symbol, horizon) =>
  client.get(`/predictions/${symbol}`, { params: { horizon } }).then((r) => r.data);

export const getTopPredictions = (horizon, limit = 10) =>
  client.get("/predictions/top", { params: { horizon, limit } }).then((r) => r.data);
