import { useAppStore } from '../store/useAppStore';

export function ClarificationDialog() {
  const result = useAppStore((s) => s.result);
  const clarificationAnswers = useAppStore((s) => s.clarificationAnswers);
  const setClarificationAnswer = useAppStore((s) => s.setClarificationAnswer);
  const submitClarifications = useAppStore((s) => s.submitClarifications);
  const error = useAppStore((s) => s.error);

  const questions = result?.clarification_questions ?? [];
  const continueDisabled = questions.some((q) => !clarificationAnswers[q.key]);

  return (
    <div className="dialog-backdrop">
      <div className="dialog blueprint elev-lg" style={{ border: '1px solid var(--color-divider)' }}>
        <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
        <div className="dialog-title">A couple quick questions</div>
        <div className="dialog-body">So we search GitHub for the right kind of project.</div>

        {questions.map((q) => (
          <div key={q.key} style={{ marginTop: 'var(--space-2)' }}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>{q.question}</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {q.options.map((opt) => (
                <label key={opt} className="radio">
                  <input
                    type="radio"
                    name={q.key}
                    checked={clarificationAnswers[q.key] === opt}
                    onChange={() => setClarificationAnswer(q.key, opt)}
                  />
                  <span className="dot" />
                  {opt}
                </label>
              ))}
            </div>
          </div>
        ))}

        {error && (
          <p style={{ color: '#b5493b', fontSize: 13 }} role="alert">
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
