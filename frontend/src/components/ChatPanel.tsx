import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/useAppStore';

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
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [chatMessages, chatPending]);

  const repoCount = result?.repositories.filter((r) => r.full_name && r.commit_sha).length ?? 0;
  if (repoCount === 0 || result?.status !== 'complete') return null;

  function handleSend() {
    const trimmed = input.trim();
    if (!trimmed) return;
    sendChatMessage(trimmed);
    setInput('');
  }

  const lastAssistant = [...chatMessages].reverse().find((m) => m.role === 'assistant');
  const followUps = lastAssistant?.follow_up_suggestions?.slice(0, 3) ?? [];

  return (
    <>
      {isChatOpen ? (
        <div
          className="card blueprint elev-lg"
          style={{
            position: 'fixed',
            zIndex: 55,
            background: 'var(--color-bg)',
            display: 'flex',
            flexDirection: 'column',
            ...(fullscreen
              ? { inset: 24, width: 'auto', height: 'auto', padding: 18 }
              : { right: 24, bottom: 88, width: 340, height: 460, padding: 14 }),
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
            <div style={{ fontSize: 13, fontWeight: 600 }}>Chat · {repoCount} repos scoped</div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button className="btn btn-icon btn-ghost" onClick={() => setFullscreen((v) => !v)} aria-label="Expand">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
                </svg>
              </button>
              <button className="btn btn-icon btn-ghost" onClick={toggleChat} aria-label="Close">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
          </div>

          <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10, padding: '10px 2px' }}>
            {chatMessages.map((m) => (
              <div
                key={m.id}
                style={{
                  alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                  background: m.role === 'user' ? 'var(--color-accent)' : 'var(--color-surface)',
                  color: m.role === 'user' ? 'var(--color-bg)' : 'inherit',
                  padding: '8px 12px',
                  fontSize: 13,
                  maxWidth: '85%',
                  whiteSpace: 'pre-wrap',
                }}
              >
                {m.content}
              </div>
            ))}
            {chatPending && (
              <div style={{ alignSelf: 'flex-start', fontSize: 13, opacity: 0.6, padding: '8px 12px' }}>Thinking…</div>
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
                <span key={label} className="tag tag-outline" style={{ cursor: 'pointer' }} onClick={() => setInput(label)}>
                  {label}
                </span>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', gap: 6 }}>
            <input
              className="input"
              placeholder="ask about the scoped repos..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="btn btn-primary btn-icon" onClick={handleSend} aria-label="Send" disabled={chatPending}>
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
