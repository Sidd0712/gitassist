import { create } from 'zustand';
import { chatAboutRepos, researchIdea } from '../services/api';
import type { AnalysisResponse, RepoChatMessage, RepoChatScopeRepository } from '../types';

type AppView = 'home' | 'loading' | 'clarify' | 'results';

interface AppState {
  view: AppView;
  idea: string;
  result: AnalysisResponse | null;
  error: string | null;
  clarificationAnswers: Record<string, string>;
  chatMessages: RepoChatMessage[];
  chatPending: boolean;
  chatError: string | null;
  isChatOpen: boolean;

  setIdea: (idea: string) => void;
  setClarificationAnswer: (key: string, value: string) => void;
  submitIdea: () => Promise<void>;
  submitClarifications: () => Promise<void>;
  sendChatMessage: (question: string) => Promise<void>;
  toggleChat: () => void;
  resetChatSession: () => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  view: 'home',
  idea: '',
  result: null,
  error: null,
  clarificationAnswers: {},
  chatMessages: [],
  chatPending: false,
  chatError: null,
  isChatOpen: false,

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

    set({
      view: 'loading',
      error: null,
      result: null,
      clarificationAnswers: {},
      chatMessages: [],
      chatPending: false,
      chatError: null,
      isChatOpen: false,
    });

    try {
      const result = await researchIdea(idea);
      if (result.status === 'error') {
        set({ view: 'results', result, error: result.error });
        get().resetChatSession();
      } else if (result.status === 'needs_clarification') {
        const clarificationAnswers = Object.fromEntries(
          result.clarification_questions.map((question) => [question.key, '']),
        );
        set({ view: 'clarify', result, error: null, clarificationAnswers });
        get().resetChatSession();
      } else {
        set({ view: 'results', result, error: null });
        get().resetChatSession();
      }
    } catch (err: any) {
      const message = err?.response?.data?.detail || err.message || 'Something went wrong';
      set({ view: 'home', error: message });
    }
  },

  submitClarifications: async () => {
    const { idea, clarificationAnswers } = get();
    if (!idea.trim()) return;

    set({ view: 'loading', error: null, chatMessages: [], chatPending: false, chatError: null, isChatOpen: false });
    try {
      const result = await researchIdea(idea, clarificationAnswers);
      if (result.status === 'needs_clarification') {
        set({ view: 'clarify', result, error: null });
        get().resetChatSession();
      } else if (result.status === 'error') {
        set({ view: 'results', result, error: result.error });
        get().resetChatSession();
      } else {
        set({ view: 'results', result, error: null });
        get().resetChatSession();
      }
    } catch (err: any) {
      const message = err?.response?.data?.detail || err.message || 'Something went wrong';
      set({ view: 'clarify', error: message });
    }
  },

  sendChatMessage: async (question) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    const { result, chatMessages } = get();
    const scopeRepositories = getChatScope(result);
    if (!result || result.status !== 'complete' || scopeRepositories.length === 0) {
      set({ chatError: 'Repo chat is only available after indexed repositories are ready.' });
      return;
    }

    const userMessage: RepoChatMessage = {
      id: createMessageId('user'),
      role: 'user',
      content: trimmedQuestion,
    };

    set((state) => ({
      chatMessages: [...state.chatMessages, userMessage],
      chatPending: true,
      chatError: null,
      isChatOpen: true,
    }));

    try {
      const response = await chatAboutRepos({
        question: trimmedQuestion,
        idea_summary: result.idea_summary,
        scope_repositories: scopeRepositories,
        messages: chatMessages.map((message) => ({
          role: message.role,
          content: message.content,
        })),
      });

      const assistantMessage: RepoChatMessage = {
        id: createMessageId('assistant'),
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        evidence_hits: response.evidence_hits,
        follow_up_suggestions: response.follow_up_suggestions,
      };

      set((state) => ({
        chatMessages: [...state.chatMessages, assistantMessage],
        chatPending: false,
        chatError: null,
      }));
    } catch (err: any) {
      const message = err?.response?.data?.detail || err.message || 'Something went wrong';
      set({ chatPending: false, chatError: message });
    }
  },

  toggleChat: () => set((state) => ({ isChatOpen: !state.isChatOpen })),

  resetChatSession: () => {
    const { result } = get();
    const hasScope = getChatScope(result).length > 0;
    set({
      chatMessages: result && result.status === 'complete' && hasScope ? [buildIntroMessage(result)] : [],
      chatPending: false,
      chatError: null,
      isChatOpen: result?.status === 'complete' && hasScope,
    });
  },

  reset: () =>
    set({
      view: 'home',
      idea: '',
      result: null,
      error: null,
      clarificationAnswers: {},
      chatMessages: [],
      chatPending: false,
      chatError: null,
      isChatOpen: false,
    }),
}));

function buildIntroMessage(result: AnalysisResponse): RepoChatMessage {
  const repoCount = getChatScope(result).length;
  const repoNames = result.repositories
    .map((repo) => repo.full_name)
    .filter(Boolean)
    .slice(0, 3)
    .join(', ');

  return {
    id: 'chat-intro',
    role: 'assistant',
    content:
      repoCount > 0
        ? `I can answer grounded questions over ${repoCount} indexed repo${repoCount === 1 ? '' : 's'} for this analysis${repoNames ? `, including ${repoNames}` : ''}. Ask about architecture, files, setup, dependencies, or how the repos implement a feature.`
        : 'Repo chat will appear here once indexed repositories are available for this analysis.',
    follow_up_suggestions: [
      'Which repository is the best end-to-end reference?',
      'Show me the most relevant files for the core architecture.',
      'What dependencies define the stack in these repos?',
    ],
    isIntro: true,
  };
}

function getChatScope(result: AnalysisResponse | null): RepoChatScopeRepository[] {
  if (!result || result.status !== 'complete') return [];
  return result.repositories
    .filter((repo) => Boolean(repo.full_name && repo.commit_sha))
    .map((repo) => ({
      full_name: repo.full_name,
      commit_sha: repo.commit_sha || '',
    }));
}

function createMessageId(prefix: 'user' | 'assistant'): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
