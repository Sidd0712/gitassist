import type { LearningStep } from '../types';

interface LearningPathProps {
  steps: LearningStep[];
}

export function LearningPath({ steps }: LearningPathProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {steps.map((step, i) => (
        <div key={step.step_number} style={{ display: 'flex', gap: 14 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 'none', width: 28 }}>
            <div
              style={{
                fontFamily: 'var(--font-heading)',
                fontSize: 22,
                fontWeight: 600,
                color: 'var(--color-accent)',
              }}
            >
              {step.step_number}
            </div>
            {i < steps.length - 1 && (
              <div style={{ flex: 1, width: 1, background: 'var(--color-divider)', margin: '6px 0' }} />
            )}
          </div>
          <div className="card blueprint elev-sm" style={{ padding: 14, marginBottom: 10, flex: 1, minWidth: 0 }}>
            <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
            <div className="card-title">{step.title}</div>
            <p className="card-body" style={{ marginTop: 4 }}>
              {step.description}
            </p>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              {step.milestone && <span className="tag tag-outline">milestone: {step.milestone}</span>}
              {step.concepts.map((c) => (
                <span key={c} className="tag tag-neutral">
                  {c}
                </span>
              ))}
            </div>
            {step.resources.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
                <span style={{ fontSize: 11, opacity: 0.6 }}>Study:</span>
                {step.resources.map((r) => (
                  <span key={r} className="tag tag-accent" style={{ fontFamily: 'var(--font-mono)' }}>
                    {r}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
