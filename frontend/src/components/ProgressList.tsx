import { useAppStore } from '../store/useAppStore';

export function ProgressList() {
  const idea = useAppStore((s) => s.idea);
  const progressSteps = useAppStore((s) => s.progressSteps);

  const trimmedIdea = idea.length > 60 ? `${idea.slice(0, 60)}…` : idea || 'your idea';

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
      <div style={{ width: '100%', maxWidth: 440 }}>
        <h6 style={{ color: 'var(--color-accent)' }}>researching</h6>
        <h3 style={{ marginBottom: 'var(--space-2)' }}>&ldquo;{trimmedIdea}&rdquo;</h3>
        <div className="card blueprint elev-sm" style={{ padding: 'var(--space-4)', marginTop: 'var(--space-3)' }}>
          <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
          {progressSteps.map((step) => (
            <div key={step.message} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', padding: '8px 0' }}>
              {step.status === 'complete' && (
                <svg
                  width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" strokeWidth="1.5"
                  style={{ flex: 'none', marginTop: 1 }}
                >
                  <circle cx="12" cy="12" r="9" />
                  <path d="M8 12l3 3 5-6" />
                </svg>
              )}
              {step.status === 'current' && (
                <svg
                  width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-text)" strokeWidth="1.5"
                  className="ga-spin" style={{ flex: 'none', marginTop: 1 }}
                >
                  <path d="M12 3a9 9 0 1 0 9 9" />
                </svg>
              )}
              {step.status === 'pending' && (
                <svg
                  width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-divider)" strokeWidth="1.5"
                  style={{ flex: 'none', marginTop: 1 }}
                >
                  <circle cx="12" cy="12" r="9" />
                </svg>
              )}
              <span
                style={{
                  fontSize: 14,
                  color: step.status === 'pending' ? 'var(--color-neutral-500)' : 'var(--color-text)',
                }}
              >
                {step.message}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
