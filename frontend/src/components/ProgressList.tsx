import { useAppStore } from '../store/useAppStore';

export function ProgressList() {
  const idea = useAppStore((s) => s.idea);
  const progressSteps = useAppStore((s) => s.progressSteps);
  const reset = useAppStore((s) => s.reset);

  const trimmedIdea = idea.length > 60 ? `${idea.slice(0, 60)}…` : idea || 'your idea';

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
      <div style={{ width: '100%', maxWidth: 520 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 16, marginBottom: 14 }}>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 500, lineHeight: 1.5, letterSpacing: 0, color: 'var(--color-text)' }}>
            $ gitassist research --idea &ldquo;{trimmedIdea}&rdquo;
          </h1>
          <button
            className="btn btn-ghost"
            style={{ fontFamily: 'var(--font-mono)', fontSize: 12, flex: 'none' }}
            onClick={reset}
            aria-label="Cancel research run"
          >
            ^C cancel
          </button>
        </div>
        <div className="log">
          {progressSteps.map((step) => (
            <div key={step.message} className={`log-line ${step.status === 'current' ? 'current' : ''} ${step.status === 'complete' ? 'complete' : ''}`}>
              <span className="marker">
                {step.status === 'complete' && (
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="ga-step-complete" style={{ position: 'absolute', inset: 0 }}>
                    <path className="ga-check" d="M3 8.5L6.5 12L13 4.5" stroke="var(--color-accent)" strokeWidth="1.6" strokeLinecap="square" />
                  </svg>
                )}
                {step.status === 'current' && <span className="marker-dot ga-pulse-marker">*</span>}
                {step.status === 'pending' && <span className="marker-dot">*</span>}
              </span>
              <span>{step.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
