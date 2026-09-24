import type {
  AnalysisResponse,
  IndexStatusResponse,
  RepoChatRequest,
  RepoChatResponse,
  RepoChatScopeRepository,
} from '../types';

const API_KEY: string = (import.meta.env.VITE_API_KEY as string) ?? '';
const BASE_URL: string = (import.meta.env.VITE_API_URL as string) ?? '';

function authHeaders(): Record<string, string> {
  return API_KEY ? { 'X-API-Key': API_KEY } : {};
}

async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const errData = await response.json();
    return (errData?.detail as string) || `HTTP ${response.status}`;
  } catch {
    return (await response.text()) || `HTTP ${response.status}`;
  }
}

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
      ...authHeaders(),
    },
    body: JSON.stringify({ idea, clarification_answers: clarificationAnswers }),
    signal,
  });

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
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
  const response = await fetch(`${BASE_URL}/health`, { headers: authHeaders() });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
}

export async function chatAboutRepos(payload: RepoChatRequest): Promise<RepoChatResponse> {
  const response = await fetch(`${BASE_URL}/research/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
}

export async function fetchIndexStatus(repositories: RepoChatScopeRepository[]): Promise<IndexStatusResponse> {
  const response = await fetch(`${BASE_URL}/research/index-status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ repositories }),
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
}
