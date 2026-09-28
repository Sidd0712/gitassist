import { useAppStore } from '../store/useAppStore';

const EXAMPLES = [
  'Realtime collaborative whiteboard',
  'Slack bot that triages GitHub issues',
  'Mobile expense tracker with receipt OCR',
];

const MAX_IDEA_LENGTH = 5000;
const MIN_IDEA_LENGTH = 10;
// Rough wrap width of the textarea at its default size — used only to give
// the "+N" stat a visible pulse as a long single-paragraph idea grows,
// since most ideas never contain a literal newline.
const CHARS_PER_WRAPPED_LINE = 60;

export function IdeaForm() {
  const idea = useAppStore((s) => s.idea);
  const setIdea = useAppStore((s) => s.setIdea);
  const submitIdea = useAppStore((s) => s.submitIdea);
  const error = useAppStore((s) => s.error);

  const trimmedLength = idea.trim().length;
  const submitDisabled = trimmedLength < MIN_IDEA_LENGTH;
  const lineCount = idea
    ? idea.split('\n').reduce((total, line) => total + Math.max(1, Math.ceil(line.length / CHARS_PER_WRAPPED_LINE)), 0)
    : 0;

  function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!submitDisabled) submitIdea();
  }

  function fillExample(label: string) {
    setIdea(`${label} — people should be able to sign in, create a project, and see live progress.`);
  }

  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
      <div style={{ width: '100%', maxWidth: 640 }}>
        <h1 className="page-title">What are you building?</h1>
        <p className="text-muted" style={{ maxWidth: '58ch', fontSize: 15.5, lineHeight: 1.65 }}>
          We&rsquo;ll search GitHub for real reference repositories, index the strongest ones, and generate a
          grounded build plan &mdash; repos, a learning path, an architecture diagram and a tech stack, all cited to
          actual code.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="card" style={{ marginTop: 'var(--space-8)', padding: 0 }}>
            <div className="patch-head">
              diff --git a/your-idea b/gitassist-report
              <br />
              --- /dev/null
              <br />
              <span className="fname">+++ b/your-idea.md</span>
            </div>
            <div className="patch-hunk">@@ -0,0 +1,{Math.max(lineCount, 1)} @@</div>
            <div className="idea-body">
              <span className="idea-mark" aria-hidden="true">+</span>
              <textarea
                id="idea-input"
                name="idea"
                className="input"
                rows={5}
                placeholder="I want to build a realtime collaborative whiteboard where multiple people can draw and sync..."
                value={idea}
                onChange={(e) => setIdea(e.target.value.slice(0, MAX_IDEA_LENGTH))}
                style={{ fontSize: 14.5, border: 'none', background: 'var(--color-accent-100)', paddingLeft: 32 }}
                autoFocus
              />
            </div>
            <div className="patch-foot">
              <span className="stat-line">
                <span className="add">+{lineCount}</span> -0 &middot; {idea.length} / {MAX_IDEA_LENGTH}
              </span>
              <button className="btn btn-primary run-btn" type="submit" disabled={submitDisabled}>
                $ gitassist research
              </button>
            </div>
          </div>
        </form>

        {error && (
          <p style={{ color: 'var(--color-danger)', fontSize: 13, marginTop: 'var(--space-2)' }} role="alert">
            {error}
          </p>
        )}

        <div className="stash" style={{ marginTop: 'var(--space-6)' }}>
          <span className="stash-label">// pick one up:</span>
          {EXAMPLES.map((label) => (
            <span
              key={label}
              className="tag tag-outline"
              style={{ cursor: 'pointer' }}
              onClick={() => fillExample(label)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  fillExample(label);
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
