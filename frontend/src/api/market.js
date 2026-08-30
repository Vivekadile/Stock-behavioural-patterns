import client from "./client";

export const getMarketOverview = () => client.get("/market/overview").then((r) => r.data);
