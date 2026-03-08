export const API_BASE =
  process.env.NEXT_PUBLIC_HALO_API_BASE?.replace(/\/$/, "") ??
  "http://127.0.0.1:8081/api/public";

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `POST ${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}
