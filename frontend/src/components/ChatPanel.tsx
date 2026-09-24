import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchIndexStatus } from '../services/api';
import { getChatScope, useAppStore } from '../store/useAppStore';
import type { Citation, IndexStatusResponse, RepoIndexState } from '../types';
import { Markdown } from './Markdown';

const STATUS_POLL_MS = 5000;
const SEARCHABLE: RepoIndexState[] = ['partial', 'completed'];
const SETTLED: RepoIndexState[] = ['completed', 'failed'];

const STATE_LABEL: Record<RepoIndexState, string> = {
  not_queued: 'not queued',
  queued: 'queued',
  indexing: 'indexing',
  partial: 'indexing…',
  completed: 'ready',
  failed: 'failed',
};

function githubLink(citation: Citation): string {
  const ref = citation.commit_sha || 'HEAD';
  const lines =
    citation.start_line != null
      ? `#L${citation.start_line}${citation.end_line != null && citation.end_line !== citation.start_line ? `-L${citation.end_line}` : ''}`
      : '';
  return `https://github.com/${citation.repo_full_name}/blob/${ref}/${citation.path}${lines}`;
}

export function ChatPanel() {
  const chatMessages = useAppStore((s) => s.chatMessages);
  const chatPending = useAppStore((s) => s.chatPending);
  const chatError = useAppStore((s) => s.chatError);
  const isChatOpen = useAppStore((s) => s.isChatOpen);
  const toggleChat = useAppStore((s) => s.toggleChat);
  const sendChatMessage = useAppStore((s) => s.sendChatMessage);
  const result = useAppStore((s) => s.result);

  const [input, setInput] = useState('');
  const [fullscreen, setFullscreen] = useState(false);
  const [status, setStatus] = useState<IndexStatusResponse | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const scope = useMemo(() => getChatScope(result), [result]);
  const scopeKey = scope.map((r) => `${r.full_name}@${r.commit_sha}`).join('|');

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [chatMessages, chatPending]);

  // Poll indexing progress until every repo is settled. Repos index on the
  // worker machine after the report is shown, so chat starts partial.
  useEffect(() => {
    if (!scope.length) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const next = await fetchIndexStatus(scope);
        if (cancelled) return;
        setStatus(next);
        if (next.repositories.every((r) => SETTLED.includes(r.state))) return;
      } catch {
        // Status is advisory; keep polling quietly.
      }
      if (!cancelled) timer = setTimeout(poll, STATUS_POLL_MS);
    };
    setStatus(null);
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  if (scope.length === 0 || result?.status !== 'complete') return null;

  const searchableCount = status?.repositories.filter((r) => SEARCHABLE.includes(r.state)).length ?? null;
  const nothingSearchable = searchableCount === 0;
  const waitingOnWorker = status && !status.worker_online && status.repositories.some((r) => !SETTLED.includes(r.state));

  function handleSend(text?: string) {
    const trimmed = (text ?? input).trim();
    if (!trimmed || chatPending) return;
    sendChatMessage(trimmed);
    setInput('');
  }

  const lastAssistant = [...chatMessages].reverse().find((m) => m.role === 'assistant');
  const followUps = lastAssistant?.follow_up_suggestions?.slice(0, 3) ?? [];

  return (
    <>
      {isChatOpen ? (
        <div
          className="card blueprint elev-lg chat-panel"
          style={{
            position: 'fixed',
            zIndex: 55,
            background: 'var(--color-bg)',
            display: 'flex',
            flexDirection: 'column',
            ...(fullscreen
              ? { inset: 24, width: 'auto', height: 'auto', padding: 18 }
              : { right: 24, bottom: 88, width: 'min(440px, calc(100vw - 32px))', height: 'min(640px, calc(100vh - 120px))', padding: 14 }),
          }}
        >
          <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderBottom: '1px solid var(--color-divider)',
              paddingBottom: 8,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600 }}>
              Code chat · {searchableCount ?? '…'}/{scope.length} repos searchable
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button className="btn btn-icon btn-ghost" onClick={() => setFullscreen((v) => !v)} aria-label={fullscreen ? 'Shrink chat' : 'Expand chat'}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
                </svg>
              </button>
              <button className="btn btn-icon btn-ghost" onClick={toggleChat} aria-label="Close chat">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
          </div>

          {status && (
            <div className="chat-status" aria-live="polite">
              {status.repositories.map((repo) => (
                <span
                  key={repo.full_name}
                  className={`chat-status-chip state-${repo.state}`}
                  title={repo.error ?? `${repo.chunk_count} code sections indexed`}
                >
                  {repo.full_name.split('/')[1]} · {STATE_LABEL[repo.state]}
                </span>
              ))}
              {waitingOnWorker && (
                <div className="chat-status-note">
                  The indexing machine is offline. Repos will be indexed as soon as it's back; chat works for any repo marked ready.
                </div>
              )}
              {!waitingOnWorker && nothingSearchable && (
                <div className="chat-status-note">Indexing the code now. Chat unlocks as soon as the first repo is searchable.</div>
              )}
            </div>
          )}

          <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, padding: '10px 2px' }}>
            {chatMessages.map((m) => (
              <div
                key={m.id}
                className={m.role === 'user' ? 'chat-msg chat-msg-user' : 'chat-msg chat-msg-assistant'}
                style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'stretch',
                  background: m.role === 'user' ? 'var(--color-accent)' : 'var(--color-surface)',
                  color: m.role === 'user' ? 'var(--color-bg)' : 'inherit',
                  padding: '8px 12px',
                  fontSize: 13,
                  maxWidth: m.role === 'user' ? '85%' : '100%',
                  whiteSpace: m.role === 'user' ? 'pre-wrap' : undefined,
                }}
              >
                {m.role === 'user' ? m.content : <Markdown text={m.content} />}
                {m.role === 'assistant' && m.citations && m.citations.length > 0 && (
                  <div className="chat-sources">
                    <div className="chat-sources-label">Sources</div>
                    {m.citations.map((c) => (
                      <a
                        key={`${c.repo_full_name}:${c.path}:${c.start_line}`}
                        href={githubLink(c)}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="chat-source"
                      >
                        {c.repo_full_name.split('/')[1]} · {c.path}
                        {c.start_line != null ? `:${c.start_line}-${c.end_line ?? c.start_line}` : ''}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {chatPending && (
              <div style={{ alignSelf: 'flex-start', fontSize: 13, opacity: 0.6, padding: '8px 12px' }}>
                Searching the code and writing an answer…
              </div>
            )}
            {chatError && (
              <div style={{ fontSize: 12, color: '#b5493b' }} role="alert">
                {chatError}
              </div>
            )}
          </div>

          {followUps.length > 0 && (
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '6px 0' }}>
              {followUps.map((label) => (
                <button
                  key={label}
                  type="button"
                  className="tag tag-outline"
                  style={{ cursor: 'pointer', textAlign: 'left' }}
                  onClick={() => handleSend(label)}
                  disabled={chatPending || nothingSearchable}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
            <textarea
              className="input"
              rows={2}
              style={{ resize: 'none', minHeight: 40, maxHeight: 120 }}
              placeholder={nothingSearchable ? 'Waiting for the first repo to finish indexing…' : 'Ask about the code (Shift+Enter for a new line)'}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              aria-label="Chat message"
            />
            <button
              className="btn btn-primary btn-icon"
              onClick={() => handleSend()}
              aria-label="Send"
              disabled={chatPending || nothingSearchable || !input.trim()}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4z" />
              </svg>
            </button>
          </div>
        </div>
      ) : (
        <button
          className="btn btn-primary btn-icon"
          onClick={toggleChat}
          style={{
            position: 'fixed',
            right: 24,
            bottom: 24,
            width: 52,
            height: 52,
            borderRadius: '50%',
            zIndex: 54,
            boxShadow: 'var(--shadow-lg)',
          }}
          aria-label="Open chat"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-4-1L3 21l2-5.5a8.5 8.5 0 1 1 16-4z" />
          </svg>
        </button>
      )}
    </>
  );
}
