import { motion } from 'framer-motion';
import { Button } from '../components/Button';
import { GlassCard } from '../components/GlassCard';
import { InputField } from '../components/InputField';
import { BackendStatus } from '../components/BackendStatus';
import { useAppStore } from '../store/useAppStore';

const exampleIdeas = [
  'an app where friends can draw together online',
  'something like Uber for tutors',
  'a marketplace for booking personal trainers with video sessions',
];

export function HomePage() {
  const { idea, setIdea, submitIdea, error } = useAppStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitIdea();
  };

  return (
    <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4 py-16">

      <BackendStatus />

      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="text-center mb-12 max-w-3xl"
      >
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 glass px-4 py-2 rounded-full text-sm text-text-secondary mb-6"
        >
          <span className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse-glow" />
          Relevance-First Research Engine
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-extrabold font-[Outfit] leading-tight mb-6">
          Turn Plain Ideas
          <br />
          <span className="gradient-text">Into Build Plans</span>
        </h1>

        <p className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto leading-relaxed">
          Describe your product in plain English. The app will infer the technical shape, ask follow-up questions only
          when a build-critical choice is unclear, and then recommend architecture, stack, and useful GitHub references.
        </p>
      </motion.div>

      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="w-full max-w-2xl space-y-4"
      >
        <GlassCard hover={false} glow="violet" className="space-y-4">
          <InputField
            id="idea-input"
            value={idea}
            onChange={setIdea}
            multiline
            rows={4}
            placeholder="Describe your project idea... for example: 'an app where friends can draw together online'"
            label="Your Project Idea"
          />
          <Button type="submit" variant="primary" size="lg" className="w-full" disabled={idea.trim().length < 10}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Plan The Build
          </Button>
        </GlassCard>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass border-red-500/30 bg-red-500/10 p-4 rounded-xl text-red-300 text-sm"
          >
            {error}
          </motion.div>
        )}
      </motion.form>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="mt-10 w-full max-w-2xl"
      >
        <p className="text-text-muted text-sm mb-3 text-center">Try an example:</p>
        <div className="flex flex-wrap gap-2 justify-center">
          {exampleIdeas.map((exampleIdea) => (
            <motion.button
              key={exampleIdea}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => setIdea(exampleIdea)}
              className="glass px-4 py-2 rounded-full text-sm text-text-secondary hover:text-white transition-colors cursor-pointer"
            >
              {exampleIdea.length > 54 ? exampleIdea.slice(0, 54) + '...' : exampleIdea}
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
