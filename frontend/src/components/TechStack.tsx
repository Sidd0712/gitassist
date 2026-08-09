import type { TechRecommendation } from '../types';

interface TechStackProps {
  items: TechRecommendation[];
}

export function TechStack({ items }: TechStackProps) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
      {items.map((t) => (
        <div key={t.name} className="card blueprint elev-sm" style={{ padding: 14 }}>
          <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
          <div className="card-kicker">{t.category}</div>
          <div className="card-title">{t.name}</div>
          <p className="card-body" style={{ marginTop: 4 }}>
            {t.why_recommended}
          </p>
          <div style={{ display: 'flex', gap: 14, marginTop: 8, fontSize: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ color: 'var(--color-accent-700)', fontWeight: 600, marginBottom: 2 }}>pros</div>
              {t.pros.map((p) => (
                <div key={p}>+ {p}</div>
              ))}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ opacity: 0.55, fontWeight: 600, marginBottom: 2 }}>cons</div>
              {t.cons.map((c) => (
                <div key={c}>– {c}</div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
