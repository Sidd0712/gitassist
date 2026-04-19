import axios from 'axios';
import type { AnalysisResponse, RepoChatRequest, RepoChatResponse } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 600_000, // 10 minutes (increased for deep indexing on slower servers)
  headers: { 'Content-Type': 'application/json' },
});

export async function researchIdea(
  idea: string,
  clarificationAnswers: Record<string, string> = {},
): Promise<AnalysisResponse> {
  const { data } = await api.post<AnalysisResponse>('/research', {
    idea,
    clarification_answers: clarificationAnswers,
  });
  return data;
}

export async function healthCheck(): Promise<{ status: string }> {
  const { data } = await api.get('/health');
  return data;
}

export async function chatAboutRepos(payload: RepoChatRequest): Promise<RepoChatResponse> {
  const { data } = await api.post<RepoChatResponse>('/research/chat', payload);
  return data;
}
