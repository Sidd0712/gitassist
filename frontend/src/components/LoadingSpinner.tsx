import { motion } from 'framer-motion';

const messages = [
  'Normalizing the idea into technical intent...',
  'Checking if any build-critical details need clarification...',
  'Searching GitHub for useful reference repositories...',
  'Inspecting READMEs, manifests, and architecture clues...',
  'Indexing the strongest repo evidence...',
  'Building recommendations for architecture and stack...',
  'Assembling the final build path...',
];

export function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-8">
      <div className="relative w-32 h-32">
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-accent-violet/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-accent-fuchsia/40"
          animate={{ rotate: -360 }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-4 rounded-full border-2 border-accent-cyan/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-0 m-auto w-4 h-4 rounded-full bg-gradient-to-r from-accent-violet to-accent-fuchsia"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      <div className="h-8 overflow-hidden">
        <motion.div
          animate={{ y: [0, -messages.length * 32] }}
          transition={{ duration: messages.length * 3, repeat: Infinity, ease: 'linear' }}
        >
          {[...messages, ...messages].map((message, index) => (
            <div key={index} className="h-8 flex items-center justify-center text-text-secondary text-sm font-medium">
              {message}
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
