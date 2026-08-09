import { useAppStore } from '../store/useAppStore';

const EXAMPLES = [
  'Realtime collaborative whiteboard',
  'Slack bot that triages GitHub issues',
  'Mobile expense tracker with receipt OCR',
];

const MAX_IDEA_LENGTH = 5000;
const MIN_IDEA_LENGTH = 10;

export function IdeaForm() {
  const idea = useAppStore((s) => s.idea);
  const setIdea = useAppStore((s) => s.setIdea);
  const submitIdea = useAppStore((s) => s.submitIdea);
  const error = useAppStore((s) => s.error);

  const trimmedLength = idea.trim().length;
  const submitDisabled = trimmedLength < MIN_IDEA_LENGTH;

  function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!submitDisabled) submitIdea();
  }

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
      <div style={{ width: '100%', maxWidth: 620 }}>
        <h6 style={{ color: 'var(--color-accent)' }}>describe your idea</h6>
        <h2 style={{ marginBottom: 'var(--space-3)' }}>What are you building?</h2>
        <p className="text-muted" style={{ maxWidth: 480 }}>
          We&rsquo;ll search GitHub for real reference repositories, index the strongest ones, and generate a
          grounded build plan &mdash; repos, a learning path, an architecture diagram and a tech stack, all cited to
          actual code.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="card blueprint elev-sm" style={{ marginTop: 'var(--space-4)', padding: 'var(--space-4)' }}>
            <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
            <textarea
              className="input"
              rows={5}
              placeholder="I want to build a realtime collaborative whiteboard where multiple people can draw and sync..."
              value={idea}
              onChange={(e) => setIdea(e.target.value.slice(0, MAX_IDEA_LENGTH))}
              style={{ fontSize: 15 }}
              autoFocus
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'var(--space-2)' }}>
              <span style={{ fontSize: 11, opacity: 0.5 }}>
                {idea.length} / {MAX_IDEA_LENGTH}
              </span>
              <button className="btn btn-primary" type="submit" disabled={submitDisabled}>
                Research this idea
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </button>
            </div>
          </div>
        </form>

        {error && (
          <p style={{ color: '#b5493b', fontSize: 13, marginTop: 'var(--space-2)' }} role="alert">
            {error}
          </p>
        )}

        <div style={{ marginTop: 'var(--space-4)', display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 12, opacity: 0.5 }}>try an example:</span>
          {EXAMPLES.map((label) => (
            <span
              key={label}
              className="tag tag-outline"
              style={{ cursor: 'pointer' }}
              onClick={() =>
                setIdea(`${label} — people should be able to sign in, create a project, and see live progress.`)
              }
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setIdea(`${label} — people should be able to sign in, create a project, and see live progress.`);
                }
              }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
