import { create } from 'zustand';
import { chatAboutRepos, researchIdea } from '../services/api';
import type { AnalysisResponse, RepoChatMessage, RepoChatScopeRepository } from '../types';

type AppView = 'home' | 'loading' | 'clarify' | 'results';
type StepStatus = 'pending' | 'current' | 'complete';
type Theme = 'light' | 'dark';

export interface IdeaHistoryEntry {
  id: string;
  title: string;
  idea: string;
  when: number;
}

const THEME_STORAGE_KEY = 'gitassist:theme';
const HISTORY_STORAGE_KEY = 'gitassist:idea-history';
const MAX_HISTORY_ENTRIES = 12;

function loadStoredTheme(): Theme | null {
  if (typeof window === 'undefined') return null;
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === 'light' || stored === 'dark' ? stored : null;
}

function systemPrefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches;
}

function loadIdeaHistory(): IdeaHistoryEntry[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveIdeaHistory(entries: IdeaHistoryEntry[]): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
}

export interface ProgressStep {
  message: string;
  status: StepStatus;
}

/** Canonical ordered list — must match backend _PROGRESS_* constants exactly. */
export const PROGRESS_STEPS: readonly string[] = [
  'Extracting intent from your idea...',
  'Searching GitHub for relevant repositories...',
  'Analysing top candidates...',
  'Indexing real-world code (this takes a moment)...',
  'Retrieving relevant code sections...',
  'Generating your research report...',
] as const;

function initialProgressSteps(): ProgressStep[] {
  return PROGRESS_STEPS.map((message, i) => ({
    message,
    status: i === 0 ? 'current' : 'pending',
  }));
}

interface AppState {
  view: AppView;
  idea: string;
  result: AnalysisResponse | null;
  error: string | null;
  clarificationAnswers: Record<string, string>;
  progressSteps: ProgressStep[];
  activeResearchController: AbortController | null;
  activeResearchRequestId: number | null;
  chatMessages: RepoChatMessage[];
  chatPending: boolean;
  chatError: string | null;
  isChatOpen: boolean;
  theme: Theme;
  ideaHistory: IdeaHistoryEntry[];

  setIdea: (idea: string) => void;
  setClarificationAnswer: (key: string, value: string) => void;
  submitIdea: () => Promise<void>;
  submitClarifications: () => Promise<void>;
  sendChatMessage: (question: string) => Promise<void>;
  toggleChat: () => void;
  resetChatSession: () => void;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  loadFromHistory: (entry: IdeaHistoryEntry) => void;
  reset: () => void;
}

function applyThemeToDocument(theme: Theme): void {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.theme = theme;
}

const initialTheme: Theme = loadStoredTheme() ?? (systemPrefersDark() ? 'dark' : 'light');
applyThemeToDocument(initialTheme);

export const useAppStore = create<AppState>((set, get) => ({
  view: 'home',
  idea: '',
  result: null,
  error: null,
  clarificationAnswers: {},
  progressSteps: initialProgressSteps(),
  activeResearchController: null,
  activeResearchRequestId: null,
  chatMessages: [],
  chatPending: false,
  chatError: null,
  isChatOpen: false,
  theme: initialTheme,
  ideaHistory: loadIdeaHistory(),

  setIdea: (idea) => set({ idea }),
  setClarificationAnswer: (key, value) =>
    set((state) => ({
      clarificationAnswers: { ...state.clarificationAnswers, [key]: value },
    })),

  submitIdea: () => _runResearch(true, set, get),
  submitClarifications: () => _runResearch(false, set, get),

  setTheme: (theme) => {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    applyThemeToDocument(theme);
    set({ theme });
  },
  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    get().setTheme(next);
  },
  loadFromHistory: (entry) => set({ idea: entry.idea }),

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
    } catch (err: unknown) {
      const message = extractErrorMessage(err);
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

  reset: () => {
    const prior = get().activeResearchController;
    if (prior) prior.abort();
    set({
      view: 'home',
      idea: '',
      result: null,
      error: null,
      clarificationAnswers: {},
      progressSteps: initialProgressSteps(),
      activeResearchController: null,
      activeResearchRequestId: null,
      chatMessages: [],
      chatPending: false,
      chatError: null,
      isChatOpen: false,
    });
  },
}));

// ─── Shared streaming run helper ──────────────────────────────────────────────

let nextResearchRequestId = 1;

async function _runResearch(
  isInitialSubmit: boolean,
  set: (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void,
  get: () => AppState,
): Promise<void> {
  const { idea, clarificationAnswers, result: previousResult } = get();
  if (!idea.trim()) return;

  if (isInitialSubmit) {
    _recordHistoryEntry(idea, set, get);
  }

  // Abort any prior in-flight stream
  const prior = get().activeResearchController;
  if (prior) prior.abort();

  const controller = new AbortController();
  const requestId = nextResearchRequestId++;

  set({
    view: 'loading',
    error: null,
    result: isInitialSubmit ? null : previousResult,
    progressSteps: initialProgressSteps(), // step 1 = current locally during preflight
    activeResearchController: controller,
    activeResearchRequestId: requestId,
    chatMessages: [],
    chatPending: false,
    chatError: null,
    isChatOpen: false,
    ...(isInitialSubmit ? { clarificationAnswers: {} } : {}),
  });

  try {
    const result = await researchIdea(
      idea,
      isInitialSubmit ? {} : clarificationAnswers,
      // onProgress — advance the step list
      (message) => {
        set((state) => {
          if (state.activeResearchRequestId !== requestId) return {};
          const idx = PROGRESS_STEPS.indexOf(message);
          if (idx === -1) return {};
          return {
            progressSteps: state.progressSteps.map((step, i) => ({
              ...step,
              status:
                i === idx
                  ? 'current'
                  : step.status === 'current'
                    ? 'complete'
                    : step.status,
            })),
          };
        });
      },
      controller.signal,
    );

    if (get().activeResearchRequestId !== requestId) return;
    set({ activeResearchController: null, activeResearchRequestId: null });

    if (result.status === 'needs_clarification') {
      if (isInitialSubmit) {
        const freshAnswers = Object.fromEntries(
          result.clarification_questions.map((q) => [q.key, '']),
        );
        set({ view: 'clarify', result, error: null, clarificationAnswers: freshAnswers });
      } else {
        // Keep existing answers — let user refine them
        set({ view: 'clarify', result, error: null });
      }
      get().resetChatSession();
    } else if (result.status === 'error') {
      set({ view: 'results', result, error: result.error ?? null });
      get().resetChatSession();
    } else {
      set({ view: 'results', result, error: null });
      get().resetChatSession();
    }
  } catch (err: unknown) {
    if (get().activeResearchRequestId !== requestId) return;
    set({ activeResearchController: null, activeResearchRequestId: null });
    if ((err as { name?: string })?.name === 'AbortError') return;
    const message = extractErrorMessage(err);
    if (isInitialSubmit) {
      set({ view: 'home', error: message });
    } else {
      set({ view: 'clarify', error: message, result: previousResult });
    }
  }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function extractErrorMessage(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return 'Something went wrong';
}

function _recordHistoryEntry(
  idea: string,
  set: (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void,
  get: () => AppState,
): void {
  const trimmed = idea.trim();
  const title = trimmed.length > 60 ? `${trimmed.slice(0, 60).trimEnd()}…` : trimmed;
  const entry: IdeaHistoryEntry = {
    id: `hist-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title,
    idea: trimmed,
    when: Date.now(),
  };
  const next = [entry, ...get().ideaHistory].slice(0, MAX_HISTORY_ENTRIES);
  saveIdeaHistory(next);
  set({ ideaHistory: next });
}

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
