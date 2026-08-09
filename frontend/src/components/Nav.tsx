import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/useAppStore';

function formatWhen(timestamp: number): string {
  const diffMs = Date.now() - timestamp;
  const minutes = Math.round(diffMs / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'yesterday' : `${days} days ago`;
}

export function Nav() {
  const theme = useAppStore((s) => s.theme);
  const toggleTheme = useAppStore((s) => s.toggleTheme);
  const ideaHistory = useAppStore((s) => s.ideaHistory);
  const loadFromHistory = useAppStore((s) => s.loadFromHistory);
  const view = useAppStore((s) => s.view);
  const [historyOpen, setHistoryOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!historyOpen) return;
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setHistoryOpen(false);
      }
    }
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, [historyOpen]);

  return (
    <div className="nav" style={{ flexWrap: 'wrap', gap: 12 }}>
      <span className="nav-brand">
        GitAssist <span style={{ color: 'var(--color-accent)' }}>AI</span>
      </span>

      <div ref={containerRef} style={{ position: 'relative', marginLeft: 'auto' }}>
        <button
          className="btn btn-secondary"
          style={{ fontSize: 13 }}
          onClick={() => setHistoryOpen((v) => !v)}
          aria-expanded={historyOpen}
        >
          History
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        {historyOpen && (
          <div
            className="card elev-md"
            style={{
              position: 'absolute',
              right: 0,
              top: 42,
              width: 260,
              zIndex: 30,
              background: 'var(--color-bg)',
              gap: 2,
              padding: 6,
              maxHeight: 320,
              overflowY: 'auto',
            }}
          >
            {ideaHistory.length === 0 ? (
              <div style={{ padding: '10px 10px', fontSize: 13, opacity: 0.6 }}>No past ideas yet</div>
            ) : (
              ideaHistory.map((h) => (
                <button
                  key={h.id}
                  onClick={() => {
                    loadFromHistory(h);
                    setHistoryOpen(false);
                  }}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '8px 10px',
                    fontSize: 13,
                    cursor: 'pointer',
                    borderRadius: 0,
                    border: 'none',
                    background: 'transparent',
                    color: 'inherit',
                    font: 'inherit',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--color-surface)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <div style={{ fontWeight: 500 }}>{h.title}</div>
                  <div style={{ fontSize: 11, opacity: 0.55 }}>{formatWhen(h.when)}</div>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      <button
        className="btn btn-icon btn-secondary"
        onClick={toggleTheme}
        aria-label="Toggle theme"
        disabled={view === 'loading'}
      >
        {theme === 'dark' ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />
          </svg>
        )}
      </button>
    </div>
  );
}
