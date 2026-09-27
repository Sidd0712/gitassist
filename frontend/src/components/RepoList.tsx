import type { Citation, RepoSearchResult } from '../types';

function humanize(value: string): string {
  const words = value.split('_');
  return words.map((word, i) => (i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word)).join('-');
}

interface RepoListProps {
  repositories: RepoSearchResult[];
  descriptions: string[];
  citations: Citation[];
  onCite: (citation: Citation) => void;
}

export function RepoList({ repositories, descriptions, citations, onCite }: RepoListProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {repositories.map((repo, i) => {
        const description = descriptions[i] || repo.description || 'No description available.';
        const repoCitations = citations.filter((c) => c.repo_full_name === repo.full_name);

        return (
          <div
            key={repo.full_name}
            className={i === 0 ? 'card blueprint elev-md' : 'card blueprint elev-sm'}
            style={{ padding: 14 }}
          >
            <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                  <a
                    className="card-title"
                    href={repo.html_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: 'inherit', textDecoration: 'none' }}
                  >
                    {repo.full_name}
                  </a>
                  {repo.reference_type && (
                    <span className="tag tag-accent-2">{humanize(repo.reference_type)}</span>
                  )}
                  {repo.evidence_type && repo.evidence_type !== 'deep_retrieval' && (
                    <span
                      className="tag tag-caution"
                      title="This description is based on the repo's README and manifest only — deep code indexing hasn't finished. Once it does, ask the code chat for answers grounded in the actual source with citations."
                    >
                      {repo.evidence_type === 'shallow_evidence'
                        ? 'overview only — not fully indexed'
                        : 'not yet reviewed'}
                    </span>
                  )}
                </div>
                <p className="card-body" style={{ marginTop: 4 }}>
                  {description}
                  {repoCitations.length > 0 && (
                    <span style={{ color: 'var(--color-accent-700)' }}>
                      {' '}
                      {repoCitations.map((_, ci) => `[${ci + 1}]`).join('')}
                    </span>
                  )}
                </p>
                <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                  {repo.topics.slice(0, 6).map((topic) => (
                    <span key={topic} className="tag tag-neutral">
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
              <div style={{ textAlign: 'right', flex: 'none', display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                <div className="card-meta" style={{ justifyContent: 'flex-end', gap: 8, fontVariantNumeric: 'tabular-nums' }}>
                  <span>★ {repo.stars.toLocaleString()}</span>
                  {repo.language && <span className="tag tag-neutral">{repo.language}</span>}
                </div>
                {i === 0 && typeof repo.fit_score === 'number' ? (
                  <span className="tag tag-accent">best match</span>
                ) : (
                  typeof repo.fit_score === 'number' && (
                    <span className="tag tag-outline" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      fit {repo.fit_score.toFixed(2)}
                    </span>
                  )
                )}
              </div>
            </div>
            {repoCitations.length > 0 && (
              <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                {repoCitations.map((c, ci) => (
                  <span
                    key={`${c.path}-${c.start_line}-${ci}`}
                    className="tag tag-accent"
                    style={{ cursor: 'pointer' }}
                    onClick={() => onCite(c)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onCite(c)}
                  >
                    evidence [{ci + 1}]
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
