import type { TechRecommendation } from '../types';
import { Typewriter } from './Typewriter';

interface TechStackProps {
  items: TechRecommendation[];
}

export function TechStack({ items }: TechStackProps) {
  const generalCount = items.filter((t) => t.supported_by.includes('general recommendation')).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {generalCount > 0 && (
        <div className="warn-row" style={{ borderTop: '1px solid var(--color-divider)' }}>
          {generalCount} of {items.length} picks are general best practices, not tied to a specific reference repo
          &mdash; the rest are directly used by the repos we found.
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 18 }}>
        {items.map((t, i) => (
          <div key={t.name} className="card ga-rise-in" style={{ padding: 0, animationDelay: `${Math.min(i, 5) * 40}ms` }}>
            <div className="file-head">
              <span className="fp">{t.name}</span>
              <span className="tag tag-neutral">{t.category}</span>
            </div>
            <div className="lines" style={{ padding: '14px 16px 16px' }}>
              <div className="dline prose">
                <span className="mk"> </span>
                <span className="tx" style={{ color: 'var(--color-text)' }}>
                  <Typewriter text={t.why_recommended} delayMs={120 + Math.min(i, 5) * 90} />
                </span>
              </div>
              {t.supported_by.length > 0 && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '10px 0 12px 20px' }}>
                  {t.supported_by.map((s) =>
                    s === 'general recommendation' ? (
                      <details className="ground-note" key={s}>
                        <summary className="tag-caution-note">general recommendation</summary>
                        <span className="ground-note-body">
                          Not backed by a specific reference repo &mdash; a general best practice.
                        </span>
                      </details>
                    ) : (
                      <details className="ground-note" key={s}>
                        <summary className="tag tag-success">{s}</summary>
                        <span className="ground-note-body">A reference repo declares this dependency.</span>
                      </details>
                    ),
                  )}
                </div>
              )}
              {t.pros.map((p) => (
                <div className="dline prose add" key={p}>
                  <span className="mk">+</span>
                  <span className="tx">{p}</span>
                </div>
              ))}
              {t.cons.map((c) => (
                <div className="dline prose" key={c}>
                  <span className="mk">-</span>
                  <span className="tx">{c}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
