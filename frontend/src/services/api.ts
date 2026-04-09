import axios from 'axios';
import type { AnalysisResponse } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 300_000,
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
