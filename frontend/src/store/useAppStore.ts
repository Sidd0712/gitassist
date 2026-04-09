import { create } from 'zustand';
import type { AnalysisResponse } from '../types';
import { researchIdea } from '../services/api';

type AppView = 'home' | 'loading' | 'clarify' | 'results';

interface AppState {
  view: AppView;
  idea: string;
  result: AnalysisResponse | null;
  error: string | null;
  clarificationAnswers: Record<string, string>;

  setIdea: (idea: string) => void;
  setClarificationAnswer: (key: string, value: string) => void;
  submitIdea: () => Promise<void>;
  submitClarifications: () => Promise<void>;
  reset: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  view: 'home',
  idea: '',
  result: null,
  error: null,
  clarificationAnswers: {},

  setIdea: (idea) => set({ idea }),
  setClarificationAnswer: (key, value) =>
    set((state) => ({
      clarificationAnswers: {
        ...state.clarificationAnswers,
        [key]: value,
      },
    })),

  submitIdea: async () => {
    const { idea } = get();
    if (!idea.trim()) return;

    set({ view: 'loading', error: null, result: null, clarificationAnswers: {} });

    try {
      const result = await researchIdea(idea);
      if (result.status === 'error') {
        set({ view: 'results', result, error: result.error });
      } else if (result.status === 'needs_clarification') {
        const clarificationAnswers = Object.fromEntries(
          result.clarification_questions.map((question) => [question.key, '']),
        );
        set({ view: 'clarify', result, error: null, clarificationAnswers });
      } else {
        set({ view: 'results', result, error: null });
      }
    } catch (err: any) {
      const message = err?.response?.data?.detail || err.message || 'Something went wrong';
      set({ view: 'home', error: message });
    }
  },

  submitClarifications: async () => {
    const { idea, clarificationAnswers } = get();
    if (!idea.trim()) return;

    set({ view: 'loading', error: null });
    try {
      const result = await researchIdea(idea, clarificationAnswers);
      if (result.status === 'needs_clarification') {
        set({ view: 'clarify', result, error: null });
      } else if (result.status === 'error') {
        set({ view: 'results', result, error: result.error });
      } else {
        set({ view: 'results', result, error: null });
      }
    } catch (err: any) {
      const message = err?.response?.data?.detail || err.message || 'Something went wrong';
      set({ view: 'clarify', error: message });
    }
  },

  reset: () => set({ view: 'home', idea: '', result: null, error: null, clarificationAnswers: {} }),
}));
