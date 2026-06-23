export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const WS_URL = API_URL.replace(/^http/, 'ws')

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, options)
  if (!response.ok) throw new Error(`API ${response.status}: ${await response.text()}`)
  return response.json()
}
