import type { LearningStep } from '../types';

interface LearningPathProps {
  steps: LearningStep[];
}

export function LearningPath({ steps }: LearningPathProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {steps.map((step) => (
        <div key={step.step_number} className="card blueprint elev-sm" style={{ padding: 14, display: 'flex', gap: 14 }}>
          <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
          <div
            style={{
              fontFamily: 'var(--font-heading)',
              fontSize: 22,
              fontWeight: 600,
              color: 'var(--color-accent)',
              flex: 'none',
              width: 28,
            }}
          >
            {step.step_number}
          </div>
          <div style={{ minWidth: 0 }}>
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
          </div>
        </div>
      ))}
    </div>
  );
}
