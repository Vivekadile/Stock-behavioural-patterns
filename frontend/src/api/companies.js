import client from "./client";

export const listCompanies = () => client.get("/companies").then((r) => r.data);

export const getCompany = (symbol) =>
  client.get(`/companies/${symbol}`).then((r) => r.data);

export const getCompanyHistory = (symbol, days = 180) =>
  client.get(`/companies/${symbol}/history`, { params: { days } }).then((r) => r.data);
