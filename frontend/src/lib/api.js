import axios from "axios";

export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const client = axios.create({
  baseURL: API_URL,
  timeout: 120000,
});

export async function postQuery(payload) {
  const { data } = await client.post("/api/query", payload);
  return data;
}

export async function fetchResults() {
  const { data } = await client.get("/api/results");
  return data;
}
