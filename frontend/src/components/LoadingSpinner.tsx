import { motion } from 'framer-motion';
import { useAppStore } from '../store/useAppStore';
import type { ProgressStep } from '../store/useAppStore';

function StepIcon({ step }: { step: ProgressStep }) {
  if (step.status === 'complete') {
    return (
      <motion.span
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-sm font-bold text-emerald-400"
      >
        {'\u2713'}
      </motion.span>
    );
  }

  if (step.status === 'current') {
    return (
      <motion.div
        className="h-6 w-6 shrink-0 rounded-full border-2 border-accent-violet border-t-transparent"
        animate={{ rotate: 360 }}
        transition={{ duration: 0.85, repeat: Infinity, ease: 'linear' }}
      />
    );
  }

  return <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 border-text-muted/20" />;
}

export function LoadingSpinner() {
  const progressSteps = useAppStore((state) => state.progressSteps);

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-8">
      <motion.div
        className="h-14 w-14 rounded-full bg-gradient-to-br from-accent-violet via-accent-fuchsia to-accent-cyan opacity-70"
        animate={{ scale: [1, 1.15, 1], opacity: [0.6, 0.9, 0.6] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
      />

      <div
        className="w-full max-w-sm space-y-4 rounded-2xl border border-border-glass bg-bg-card px-6 py-5 backdrop-blur-sm"
        role="status"
        aria-label="Research progress"
      >
        {progressSteps.map((step) => (
          <motion.div
            key={step.message}
            initial={false}
            animate={{ opacity: step.status === 'pending' ? 0.4 : 1 }}
            transition={{ duration: 0.3 }}
            className="flex items-center gap-3"
          >
            <StepIcon step={step} />
            <span
              className={[
                'text-sm leading-snug transition-colors duration-300',
                step.status === 'complete'
                  ? 'text-text-secondary line-through decoration-text-muted/40'
                  : step.status === 'current'
                    ? 'font-medium text-text-primary'
                    : 'text-text-muted',
              ].join(' ')}
            >
              {step.message}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
