import type { LearningStep } from '../types';
import { Typewriter } from './Typewriter';

interface LearningPathProps {
  steps: LearningStep[];
}

export function LearningPath({ steps }: LearningPathProps) {
  return (
    <div className="card" style={{ padding: 0 }}>
      <div className="file-head">
        <span className="fp">learning-path.md</span>
        <span className="stat">{steps.length} steps</span>
      </div>
      {steps.map((step, i) => (
        <div
          key={step.step_number}
          className="ga-rise-in"
          style={{
            borderTop: i === 0 ? 'none' : '1px solid var(--color-divider)',
            animationDelay: `${Math.min(i, 5) * 40}ms`,
          }}
        >
          <div className="hunk-head">
            <span className="hunk">@@ step {step.step_number} @@</span>
            <h3 className="hunk-title">{step.title}</h3>
          </div>
          <div className="lines" style={{ padding: '8px 16px 20px' }}>
            <div className="dline prose">
              <span className="mk"> </span>
              <span className="tx">
                <Typewriter text={step.description} delayMs={120 + Math.min(i, 5) * 90} />
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '12px 0 0 20px' }}>
              {step.milestone && <span className="tag tag-outline">milestone: {step.milestone}</span>}
              {step.concepts.map((c) => (
                <span key={c} className="tag tag-neutral">
                  {c}
                </span>
              ))}
            </div>
            {step.resources.length > 0 && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '10px 0 0 20px', alignItems: 'center' }}>
                <span className="stash-label">Study:</span>
                {step.resources.map((r) => (
                  <span key={r} className="tag tag-accent">
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
