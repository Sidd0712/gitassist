import { motion } from 'framer-motion';
import { Button } from '../components/Button';
import { GlassCard } from '../components/GlassCard';
import { useAppStore } from '../store/useAppStore';

export function ClarificationPage() {
  const {
    idea,
    result,
    clarificationAnswers,
    setClarificationAnswer,
    submitClarifications,
    reset,
    error,
  } = useAppStore();

  if (!result) return null;

  const questions = result.clarification_questions;
  const canSubmit = questions.every((question) => clarificationAnswers[question.key]?.trim());

  return (
    <div className="relative z-10 min-h-screen px-4 py-12 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-8"
      >
        <Button variant="ghost" size="sm" onClick={reset}>
          Back
        </Button>
        <h2 className="text-2xl font-bold font-[Outfit] gradient-text">Clarify The Build</h2>
        <div className="w-16" />
      </motion.div>

      <GlassCard glow="violet" className="mb-8">
        <h3 className="text-sm text-text-muted uppercase tracking-wider mb-2">Your Idea</h3>
        <p className="text-text-primary text-lg">{result.idea_summary || idea}</p>
        <p className="text-text-secondary text-sm mt-3">
          A few build-critical details are still ambiguous. Answer these so we can recommend the right architecture,
          tech stack, and reference repositories.
        </p>
      </GlassCard>

      <div className="space-y-5">
        {questions.map((question, index) => (
          <GlassCard key={question.key} delay={0.08 * index} className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold font-[Outfit] text-text-primary">{question.question}</h3>
              <p className="text-sm text-text-secondary mt-1">{question.reason}</p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {question.options.map((option) => {
                const selected = clarificationAnswers[question.key] === option;
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setClarificationAnswer(question.key, option)}
                    className={[
                      'rounded-xl border px-4 py-3 text-left transition-all duration-300',
                      selected
                        ? 'border-accent-violet bg-accent-violet/15 text-white shadow-[0_0_20px_rgba(139,92,246,0.18)]'
                        : 'border-white/10 bg-white/[0.03] text-text-secondary hover:border-accent-cyan/30 hover:text-white',
                    ].join(' ')}
                  >
                    {option}
                  </button>
                );
              })}
            </div>
          </GlassCard>
        ))}
      </div>

      {error && (
        <div className="glass border-red-500/30 bg-red-500/10 p-4 rounded-xl text-red-300 text-sm mt-6">
          {error}
        </div>
      )}

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="mt-8">
        <Button variant="primary" size="lg" className="w-full" onClick={submitClarifications} disabled={!canSubmit}>
          Continue To Recommendations
        </Button>
      </motion.div>
    </div>
  );
}
