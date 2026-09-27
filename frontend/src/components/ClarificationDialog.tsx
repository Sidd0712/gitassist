import { useRef } from 'react';
import { useAppStore } from '../store/useAppStore';
import { useOverlayA11y } from '../hooks/useOverlayA11y';

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
    <div className="dialog-backdrop">
      <div
        ref={dialogRef}
        className="dialog blueprint elev-lg"
        style={{ border: '1px solid var(--color-divider)' }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="clarify-dialog-title"
        tabIndex={-1}
      >
        <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
        <div className="dialog-title" id="clarify-dialog-title">A couple quick questions</div>
        <div className="dialog-body">
          We ground every repo we pick in real evidence, not a guess — a couple more details narrows the search to the right kind of project.
        </div>

        {questions.map((q) => (
          <div key={q.key} style={{ marginTop: 'var(--space-3)' }}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>{q.question}</div>
            <div className="seg">
              {q.options.map((opt) => (
                <label key={opt} className="seg-opt">
                  <input
                    id={`${q.key}-${opt}`}
                    type="radio"
                    name={q.key}
                    checked={clarificationAnswers[q.key] === opt}
                    onChange={() => setClarificationAnswer(q.key, opt)}
                  />
                  {opt}
                </label>
              ))}
            </div>
          </div>
        ))}

        {error && (
          <p style={{ color: 'var(--color-danger)', fontSize: 13 }} role="alert">
            {error}
          </p>
        )}

        <div className="dialog-actions">
          <button className="btn btn-primary" onClick={submitClarifications} disabled={continueDisabled}>
            Continue research
          </button>
        </div>
      </div>
    </div>
  );
}
