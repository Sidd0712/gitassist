import axios from 'axios';
import type { AnalysisResponse, RepoChatRequest, RepoChatResponse } from '../types';

const API_KEY: string = (import.meta.env.VITE_API_KEY as string) ?? '';

// axios instance kept for health + chat (non-streaming endpoints)
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 600_000,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
});

const BASE_URL: string = (import.meta.env.VITE_API_URL as string) ?? '';

/**
 * Stream a research request via SSE.
 *
 * @param idea                 The user's project idea.
 * @param clarificationAnswers Answers from the clarification step (optional).
 * @param onProgress           Callback invoked for each progress message received.
 * @param signal               AbortSignal to cancel the in-flight request.
 */
export async function researchIdea(
  idea: string,
  clarificationAnswers: Record<string, string> = {},
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<AnalysisResponse> {
  const response = await fetch(`${BASE_URL}/research`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
    },
    body: JSON.stringify({ idea, clarification_answers: clarificationAnswers }),
    signal,
  });

  if (!response.ok) {
    let errorMessage: string;
    try {
      const errData = await response.json();
      errorMessage = (errData?.detail as string) || `HTTP ${response.status}`;
    } catch {
      errorMessage = (await response.text()) || `HTTP ${response.status}`;
    }
    throw new Error(errorMessage);
  }

  if (!response.body) {
    throw new Error('Streaming response did not include a body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      const trailingEvent = parseSseEvent(buffer);
      if (trailingEvent?.type === 'result' && trailingEvent.data) {
        return trailingEvent.data;
      }
      if (trailingEvent?.type === 'error') {
        throw new Error(trailingEvent.message ?? 'Stream error from server');
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by blank lines.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? ''; // keep any incomplete trailing block

    for (const block of blocks) {
      const event = parseSseEvent(block);
      if (!event) continue;

      if (event.type === 'progress' && event.message) {
        onProgress?.(event.message);
      } else if (event.type === 'result' && event.data) {
        return event.data;
      } else if (event.type === 'error') {
        throw new Error(event.message ?? 'Stream error from server');
      }
    }
  }

  throw new Error('Stream ended without a result event');
}

function parseSseEvent(block: string): { type: string; message?: string; data?: AnalysisResponse } | null {
  const dataLines = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(line.startsWith('data: ') ? 6 : 5));

  if (!dataLines.length) return null;

  try {
    return JSON.parse(dataLines.join('\n'));
  } catch {
    return null;
  }
}

export async function healthCheck(): Promise<{ status: string }> {
  const { data } = await api.get('/health');
  return data;
}

export async function chatAboutRepos(payload: RepoChatRequest): Promise<RepoChatResponse> {
  const { data } = await api.post<RepoChatResponse>('/research/chat', payload);
  return data;
}
