export interface CitationDetail {
  repo_full_name: string;
  path: string;
  start_line: number | null;
  end_line: number | null;
  reason: string;
  snippet?: string;
}

interface CitationPopoverProps {
  citation: CitationDetail | null;
  repoUrl?: string;
  onClose: () => void;
}

export function CitationPopover({ citation, repoUrl, onClose }: CitationPopoverProps) {
  if (!citation) return null;

  const lineRange =
    citation.start_line != null && citation.end_line != null ? `lines ${citation.start_line}–${citation.end_line}` : null;
  const githubLink =
    repoUrl && citation.start_line != null
      ? `${repoUrl}/blob/HEAD/${citation.path}#L${citation.start_line}${citation.end_line ? `-L${citation.end_line}` : ''}`
      : repoUrl
        ? `${repoUrl}/blob/HEAD/${citation.path}`
        : null;

  return (
    <div
      style={{ position: 'fixed', left: '50%', bottom: 24, transform: 'translateX(-50%)', zIndex: 50, width: 'min(520px, 92vw)' }}
      className="card blueprint elev-lg"
    >
      <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="tag tag-accent">
          {citation.repo_full_name} · {citation.path}
        </span>
        <button className="btn btn-icon btn-ghost" onClick={onClose} aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      </div>
      {lineRange && <div style={{ fontSize: 11, opacity: 0.6, marginTop: 4 }}>{lineRange}</div>}
      {citation.reason && <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>{citation.reason}</div>}
      {citation.snippet ? (
        <pre
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            background: 'var(--color-surface)',
            padding: 10,
            margin: '8px 0 0',
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
          }}
        >
          {citation.snippet}
        </pre>
      ) : githubLink ? (
        <a href={githubLink} target="_blank" rel="noreferrer" style={{ display: 'inline-block', marginTop: 8, fontSize: 13 }}>
          View this file on GitHub →
        </a>
      ) : null}
    </div>
  );
}
