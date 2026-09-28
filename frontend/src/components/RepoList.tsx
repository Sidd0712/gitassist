import type { Citation, RepoSearchResult } from '../types';
import { humanize } from '../utils';
import { Typewriter } from './Typewriter';

interface RepoListProps {
  repositories: RepoSearchResult[];
  descriptions: string[];
  citations: Citation[];
  onCite: (citation: Citation) => void;
}

export function RepoList({ repositories, descriptions, citations, onCite }: RepoListProps) {
  const shallowCount = repositories.filter((r) => r.evidence_type && r.evidence_type !== 'deep_retrieval').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {shallowCount > 0 && (
        <div className="warn-row" style={{ borderTop: '1px solid var(--color-divider)' }}>
          {repositories.length - shallowCount} of {repositories.length} repos are fully indexed below &mdash; the rest
          show a README-based overview for now and unlock cited chat answers once indexing finishes.
        </div>
      )}
      {repositories.map((repo, i) => {
        const description = descriptions[i] || repo.description || 'No description available.';
        const repoCitations = citations.filter((c) => c.repo_full_name === repo.full_name);
        const isTop = i === 0;

        return (
          <div
            key={repo.full_name}
            className="card ga-rise-in"
            style={{ padding: 0, animationDelay: `${Math.min(i, 5) * 40}ms` }}
          >
            <div className="file-head">
              <a href={repo.html_url} target="_blank" rel="noreferrer" className="fp" style={{ textDecoration: 'none' }}>
                {repo.full_name}
              </a>
              {repo.reference_type && <span className="tag tag-accent-2">{humanize(repo.reference_type)}</span>}
              {repo.evidence_type && repo.evidence_type !== 'deep_retrieval' && (
                <details className="ground-note">
                  <summary className="tag-caution-note">
                    {repo.evidence_type === 'shallow_evidence' ? 'overview only' : 'not yet reviewed'}
                  </summary>
                  <span className="ground-note-body">
                    This description is based on the repo&rsquo;s README and manifest only &mdash; deep code indexing
                    hasn&rsquo;t finished. Once it does, ask the code chat for answers grounded in the actual source
                    with citations.
                  </span>
                </details>
              )}
              <span className="stat">
                {isTop && typeof repo.fit_score === 'number' ? (
                  <span className="tag tag-accent">best match</span>
                ) : typeof repo.fit_score === 'number' ? (
                  <span className="stat-line">
                    <span className="add">+{Math.round(repo.fit_score * 100)}</span>
                  </span>
                ) : null}
              </span>
            </div>
            <div className="lines" style={{ padding: '14px 16px' }}>
              <div className={isTop ? 'dline prose add' : 'dline prose'}>
                <span className="mk">{isTop ? '+' : ' '}</span>
                <span className="tx">
                  <Typewriter
                    text={description}
                    delayMs={120 + Math.min(i, 5) * 90}
                    after={
                      repoCitations.length > 0 && (
                        <span style={{ color: 'var(--color-accent)', fontFamily: 'var(--font-mono)', fontSize: 12.5 }}>
                          {' '}
                          {repoCitations.map((_, ci) => `[${ci + 1}]`).join('')}
                        </span>
                      )
                    }
                  />
                </span>
              </div>
              <div className="dline meta" style={{ marginTop: 6 }}>
                <span className="mk"> </span>
                <span className="tx" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  ★ {repo.stars.toLocaleString()}
                  {repo.language ? ` · ${repo.language}` : ''}
                  {repo.topics.length > 0 ? ` · ${repo.topics.slice(0, 6).join(', ')}` : ''}
                </span>
              </div>
            </div>
            {repoCitations.length > 0 && (
              <div style={{ display: 'flex', gap: 8, padding: '0 16px 16px 36px', flexWrap: 'wrap' }}>
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
