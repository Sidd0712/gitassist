import { useRef } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useOverlayA11y } from '../hooks/useOverlayA11y';
import { humanize } from '../utils';

export function ClarificationDialog() {
  const result = useAppStore((s) => s.result);
  const clarificationAnswers = useAppStore((s) => s.clarificationAnswers);
  const setClarificationAnswer = useAppStore((s) => s.setClarificationAnswer);
  const submitClarifications = useAppStore((s) => s.submitClarifications);
  const error = useAppStore((s) => s.error);
  const reset = useAppStore((s) => s.reset);

  const questions = result?.clarification_questions ?? [];
  const continueDisabled = questions.some((q) => !clarificationAnswers[q.key]);

  const dialogRef = useRef<HTMLDivElement>(null);
  useOverlayA11y(dialogRef, reset, { active: true, trapFocus: true });

  return (
    <div className="dialog-backdrop ga-fade-in">
      <div
        ref={dialogRef}
        className="card ga-pop-in"
        style={{ width: 'min(480px, 92vw)', padding: 0 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="clarify-dialog-title"
        tabIndex={-1}
      >
        <div className="file-head">
          <span className="fp" id="clarify-dialog-title">
            +++ b/your-idea.md
          </span>
          <span className="stat">clarify</span>
        </div>
        <div className="hunk" style={{ padding: '10px 14px 0' }}>
          @@ narrowing the search @@
        </div>
        <div style={{ padding: '4px 14px 12px', fontSize: 13, color: 'var(--color-text-dim)' }}>
          We ground every repo we pick in real evidence, not a guess &mdash; a couple more details narrows the search
          to the right kind of project.
        </div>

        {questions.map((q) => (
          <div key={q.key} style={{ padding: '0 14px 14px' }}>
            <div className="dline">
              <span className="mk">?</span>
              <span className="tx" style={{ color: 'var(--color-text)', fontWeight: 500 }}>
                {q.question}
              </span>
            </div>
            <div className="seg" style={{ marginTop: 8, marginLeft: 22 }}>
              {q.options.map((opt) => (
                <label key={opt} className="seg-opt">
                  <input
                    id={`${q.key}-${opt}`}
                    type="radio"
                    name={q.key}
                    checked={clarificationAnswers[q.key] === opt}
                    onChange={() => setClarificationAnswer(q.key, opt)}
                  />
                  {humanize(opt)}
                </label>
              ))}
            </div>
          </div>
        ))}

        {error && (
          <div className="warn-row" role="alert" style={{ margin: '0 14px 14px', borderTop: '1px solid var(--color-divider)' }}>
            {error}
          </div>
        )}

        <div className="patch-foot">
          <span className="stat-line">
            {questions.length > 1 ? `answer all ${questions.length} to continue the diff` : 'answer to continue the diff'}
          </span>
          <button className="btn btn-primary run-btn" onClick={submitClarifications} disabled={continueDisabled}>
            $ gitassist continue
          </button>
        </div>
      </div>
    </div>
  );
}
