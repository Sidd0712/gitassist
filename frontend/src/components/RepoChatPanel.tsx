import { useEffect, useRef, useState } from 'react';
import { Button } from './Button';
import { GlassCard } from './GlassCard';
import { useAppStore } from '../store/useAppStore';
import type { AnalysisResponse, RepoChatMessage } from '../types';

interface RepoChatPanelProps {
  result: AnalysisResponse;
}

function ChatMessageCard({
  message,
  onSuggestionClick,
}: {
  message: RepoChatMessage;
  onSuggestionClick: (value: string) => void;
}) {
  const [showEvidence, setShowEvidence] = useState(Boolean(message.isIntro));
  const isAssistant = message.role === 'assistant';
  const hasEvidence = Boolean((message.citations && message.citations.length) || (message.evidence_hits && message.evidence_hits.length));

  return (
    <div className={`flex ${isAssistant ? 'justify-start' : 'justify-end'}`}>
      <div
        className={[
          'max-w-[92%] rounded-2xl px-4 py-3',
          isAssistant
            ? 'bg-white/[0.04] border border-white/10 text-text-primary'
            : 'bg-gradient-to-r from-accent-violet to-accent-fuchsia text-white shadow-[0_0_30px_rgba(139,92,246,0.18)]',
        ].join(' ')}
      >
        <p className="text-sm leading-6 whitespace-pre-wrap">{message.content}</p>

        {message.follow_up_suggestions && message.follow_up_suggestions.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.follow_up_suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => onSuggestionClick(suggestion)}
                className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs text-text-secondary transition hover:border-accent-cyan/40 hover:text-white"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {isAssistant && hasEvidence && (
          <div className="mt-3">
            <button
              type="button"
              onClick={() => setShowEvidence((current) => !current)}
              className="text-xs text-accent-cyan transition hover:text-white"
            >
              {showEvidence ? 'Hide evidence' : 'Show evidence'}
            </button>
            {showEvidence && (
              <div className="mt-3 space-y-3">
                {message.citations && message.citations.length > 0 && (
                  <div>
                    <p className="mb-2 text-[11px] uppercase tracking-[0.24em] text-text-muted">Citations</p>
                    <div className="space-y-2">
                      {message.citations.map((citation) => (
                        <div key={`${citation.repo_full_name}:${citation.path}:${citation.start_line ?? 'na'}`} className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                          <div className="text-xs font-semibold text-text-primary">{citation.repo_full_name}</div>
                          <div className="mt-1 text-xs text-text-secondary">
                            {citation.path}
                            {formatLineSpan(citation.start_line, citation.end_line)}
                          </div>
                          {citation.reason && <div className="mt-1 text-xs text-text-muted">{citation.reason}</div>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {message.evidence_hits && message.evidence_hits.length > 0 && (
                  <div>
                    <p className="mb-2 text-[11px] uppercase tracking-[0.24em] text-text-muted">Retrieved Chunks</p>
                    <div className="space-y-2">
                      {message.evidence_hits.map((hit) => (
                        <div key={`${hit.repo_full_name}:${hit.path}:${hit.start_line ?? 'na'}:${hit.score}`} className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-xs font-semibold text-text-primary">{hit.repo_full_name}</div>
                            <div className="text-[11px] text-text-muted">score {(hit.score * 100).toFixed(0)}%</div>
                          </div>
                          <div className="mt-1 text-xs text-text-secondary">
                            {hit.path}
                            {formatLineSpan(hit.start_line, hit.end_line)}
                          </div>
                          {hit.reason && <div className="mt-1 text-xs text-text-muted">{hit.reason}</div>}
                          {hit.snippet && <p className="mt-2 text-xs leading-5 text-text-secondary">{hit.snippet}</p>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function RepoChatPanelBody({
  result,
  onClose,
}: {
  result: AnalysisResponse;
  onClose?: () => void;
}) {
  const chatMessages = useAppStore((state) => state.chatMessages);
  const chatPending = useAppStore((state) => state.chatPending);
  const chatError = useAppStore((state) => state.chatError);
  const sendChatMessage = useAppStore((state) => state.sendChatMessage);
  const [draft, setDraft] = useState('');
  const endRef = useRef<HTMLDivElement | null>(null);

  const scopedRepos = result.repositories.filter((repo) => Boolean(repo.full_name && repo.commit_sha));
  const hasScope = result.status === 'complete' && scopedRepos.length > 0;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [chatMessages, chatPending]);

  const submit = async () => {
    if (!draft.trim() || chatPending || !hasScope) return;
    const nextQuestion = draft;
    setDraft('');
    await sendChatMessage(nextQuestion);
  };

  return (
    <GlassCard hover={false} glow="violet" className="flex h-full min-h-[70vh] flex-col gap-4 p-0 overflow-hidden">
      <div className="border-b border-white/10 px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.28em] text-text-muted">Groq + Repo RAG</div>
            <h3 className="mt-2 font-[Outfit] text-xl text-text-primary">Repo Chat</h3>
            <p className="mt-2 text-sm leading-6 text-text-secondary">
              Ask about architecture, files, setup flow, or dependencies across {scopedRepos.length} indexed repo
              {scopedRepos.length === 1 ? '' : 's'}.
            </p>
          </div>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-text-secondary transition hover:text-white"
            >
              Close
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {!hasScope && (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-5 text-sm leading-6 text-text-secondary">
            Repo chat will unlock once this analysis has indexed repositories with commit snapshots.
          </div>
        )}

        {chatMessages.map((message) => (
          <ChatMessageCard key={message.id} message={message} onSuggestionClick={setDraft} />
        ))}

        {chatPending && (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-text-secondary">
            Searching indexed files and asking Groq for a grounded answer...
          </div>
        )}

        {chatError && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {chatError}
          </div>
        )}

        <div ref={endRef} />
      </div>

      <div className="border-t border-white/10 px-5 py-4">
        <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void submit();
              }
            }}
            placeholder={hasScope ? 'Ask about the repos or the retrieved evidence...' : 'Repo chat is waiting for indexed repositories.'}
            rows={3}
            disabled={!hasScope || chatPending}
            className="min-h-[86px] w-full resize-none bg-transparent px-1 py-1 text-sm text-text-primary outline-none placeholder:text-text-muted disabled:cursor-not-allowed"
          />
          <div className="mt-3 flex items-center justify-between gap-3">
            <div className="text-xs text-text-muted">Press Enter to send, Shift+Enter for a new line.</div>
            <Button type="button" size="sm" onClick={() => void submit()} disabled={!draft.trim() || !hasScope} isLoading={chatPending}>
              Ask Groq
            </Button>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

export function RepoChatPanel({ result }: RepoChatPanelProps) {
  const isChatOpen = useAppStore((state) => state.isChatOpen);
  const toggleChat = useAppStore((state) => state.toggleChat);

  if (!isChatOpen) return null;

  return (
    <>
      <aside className="hidden lg:block">
        <div className="sticky top-6">
          <RepoChatPanelBody result={result} />
        </div>
      </aside>

      <div className="fixed inset-0 z-30 lg:hidden">
        <button
          type="button"
          aria-label="Close repo chat"
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          onClick={toggleChat}
        />
        <div className="absolute inset-y-0 right-0 w-full max-w-md p-3">
          <RepoChatPanelBody result={result} onClose={toggleChat} />
        </div>
      </div>
    </>
  );
}

function formatLineSpan(startLine: number | null, endLine: number | null): string {
  if (startLine == null && endLine == null) return '';
  if (startLine != null && endLine != null && startLine !== endLine) {
    return `:${startLine}-${endLine}`;
  }
  return `:${startLine ?? endLine}`;
}
