import { motion } from 'framer-motion';
import { Button } from '../components/Button';
import { GlassCard } from '../components/GlassCard';
import { MermaidDiagram } from '../components/MermaidDiagram';
import { RepoChatPanel } from '../components/RepoChatPanel';
import { useAppStore } from '../store/useAppStore';
import type { AnalysisResponse } from '../types';

function IntentPanel({ result }: { result: AnalysisResponse }) {
  const chips = [...result.keywords.capabilities, ...result.keywords.frameworks, ...result.keywords.languages].slice(0, 12);

  return (
    <GlassCard glow="fuchsia" className="space-y-4">
      <div>
        <h3 className="mb-2 text-sm uppercase tracking-wider text-text-muted">Resolved Idea</h3>
        <p className="text-lg text-text-primary">{result.idea_summary}</p>
      </div>

      {chips.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {chips.map((chip) => (
            <span
              key={chip}
              className="rounded-full border border-accent-violet/20 bg-accent-violet/15 px-3 py-1 text-sm text-accent-violet"
            >
              {chip}
            </span>
          ))}
        </div>
      )}

      {result.assumptions.length > 0 && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-text-primary">Assumptions</h4>
          <div className="flex flex-wrap gap-2">
            {result.assumptions.map((assumption) => (
              <span key={assumption} className="rounded-xl bg-white/[0.03] px-3 py-2 text-sm text-text-secondary">
                {assumption}
              </span>
            ))}
          </div>
        </div>
      )}
    </GlassCard>
  );
}

function RepoCards({ repos, descriptions }: { repos: AnalysisResponse['repositories']; descriptions: string[] }) {
  if (!repos.length) return null;

  return (
    <div>
      <h3 className="mb-4 text-xl font-semibold font-[Outfit]">Useful Reference Repositories</h3>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {repos.map((repo, index) => (
          <GlassCard key={repo.full_name} delay={0.08 * index} glow={index === 0 ? 'violet' : 'none'}>
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <a
                  href={repo.html_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-base font-semibold text-accent-cyan hover:underline"
                >
                  {repo.full_name}
                </a>
                <p className="mt-2 text-sm text-text-secondary">{repo.fit_summary || repo.description || 'Reference repo'}</p>
              </div>
              <div className="shrink-0 text-right">
                <div className="text-sm text-yellow-400">* {repo.stars.toLocaleString()}</div>
                {repo.fit_score !== undefined && (
                  <div className="mt-1 text-xs text-text-muted">fit {(repo.fit_score * 100).toFixed(0)}%</div>
                )}
              </div>
            </div>

            <div className="mb-3 flex flex-wrap gap-2">
              {repo.reference_type && (
                <span className="rounded-full bg-accent-fuchsia/15 px-2.5 py-1 text-xs uppercase tracking-wider text-accent-fuchsia">
                  {repo.reference_type.replace('_', ' ')}
                </span>
              )}
              {repo.language && <span className="rounded-full bg-white/5 px-2.5 py-1 text-xs text-text-muted">{repo.language}</span>}
              {repo.covered_primary?.map((capability) => (
                <span key={capability} className="rounded-full bg-accent-cyan/10 px-2.5 py-1 text-xs text-accent-cyan">
                  covers {capability}
                </span>
              ))}
            </div>

            {descriptions[index] && <p className="text-sm leading-relaxed text-text-primary">{descriptions[index]}</p>}
            {repo.missing_primary && repo.missing_primary.length > 0 && (
              <p className="mt-3 text-xs text-text-muted">Missing: {repo.missing_primary.join(', ')}</p>
            )}
          </GlassCard>
        ))}
      </div>
    </div>
  );
}

function BuildPath({ steps }: { steps: AnalysisResponse['learning_path'] }) {
  if (!steps.length) return null;

  return (
    <GlassCard>
      <h3 className="mb-5 text-xl font-semibold font-[Outfit]">Build Path</h3>
      <div className="space-y-5">
        {steps.map((step) => (
          <div key={step.step_number} className="flex gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-r from-accent-violet to-accent-fuchsia font-semibold text-white">
              {step.step_number}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs uppercase tracking-wider text-text-muted">{step.milestone}</p>
              <h4 className="mt-1 font-semibold text-text-primary">{step.title}</h4>
              <p className="mt-2 text-sm text-text-secondary">{step.description}</p>
              {step.concepts.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {step.concepts.map((concept) => (
                    <span key={concept} className="rounded-full bg-white/5 px-2.5 py-1 text-xs text-text-muted">
                      {concept}
                    </span>
                  ))}
                </div>
              )}
              {step.resources.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-3">
                  {step.resources.map((resource) => (
                    <a
                      key={resource}
                      href={resource.startsWith('http') ? resource : '#'}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-accent-cyan hover:underline"
                    >
                      {resource.length > 54 ? `${resource.slice(0, 54)}...` : resource}
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

function TechStack({ stack }: { stack: AnalysisResponse['tech_stack'] }) {
  if (!stack.length) return null;

  return (
    <GlassCard>
      <h3 className="mb-4 text-xl font-semibold font-[Outfit]">Recommended Tech Stack</h3>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {stack.map((tech) => (
          <div key={tech.name} className="glass rounded-xl p-4">
            <div className="mb-2 flex items-center gap-2">
              <span className="font-semibold text-text-primary">{tech.name}</span>
              <span className="rounded bg-accent-violet/20 px-2 py-0.5 text-[10px] uppercase tracking-wider text-accent-violet">
                {tech.category}
              </span>
            </div>
            <p className="text-sm text-text-secondary">{tech.why_recommended}</p>
            {tech.supported_by.length > 0 && (
              <p className="mt-2 text-xs text-text-muted">Supported by: {tech.supported_by.join(', ')}</p>
            )}
            {tech.pros.length > 0 && <div className="mt-3 text-xs text-green-300/85">{tech.pros.join(' | ')}</div>}
            {tech.cons.length > 0 && <div className="mt-2 text-xs text-red-300/70">{tech.cons.join(' | ')}</div>}
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

export function ResultsPage() {
  const result = useAppStore((state) => state.result);
  const reset = useAppStore((state) => state.reset);
  const error = useAppStore((state) => state.error);
  const isChatOpen = useAppStore((state) => state.isChatOpen);
  const toggleChat = useAppStore((state) => state.toggleChat);

  if (!result) {
    return (
      <div className="relative z-10 mx-auto flex min-h-screen max-w-3xl items-center justify-center px-4 py-12">
        <GlassCard glow="violet" className="w-full max-w-xl text-center">
          <h2 className="text-2xl font-bold font-[Outfit] text-text-primary">Results Unavailable</h2>
          <p className="mt-3 text-sm leading-6 text-text-secondary">
            The results view opened without a loaded analysis. Go back and run the research again.
          </p>
          <div className="mt-6">
            <Button variant="secondary" onClick={reset}>
              Back To Home
            </Button>
          </div>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className={`relative z-10 min-h-screen px-4 py-12 ${isChatOpen ? 'mx-auto max-w-7xl' : 'mx-auto max-w-5xl'}`}>
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 flex flex-wrap items-center justify-between gap-3"
      >
        <Button variant="ghost" size="sm" onClick={reset}>
          Back
        </Button>
        <h2 className="text-center text-2xl font-bold font-[Outfit] gradient-text">Build Recommendations</h2>
        <Button variant={isChatOpen ? 'secondary' : 'ghost'} size="sm" onClick={toggleChat}>
          Repo Chat
        </Button>
      </motion.div>

      {error && <div className="glass mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">{error}</div>}

      <div className={isChatOpen ? 'lg:grid lg:grid-cols-[minmax(0,1fr)_380px] lg:gap-8' : ''}>
        <div className="space-y-8">
          <IntentPanel result={result} />
          <RepoCards repos={result.repositories} descriptions={result.repo_descriptions} />
          <MermaidDiagram chart={result.architecture_diagram} />
          <BuildPath steps={result.learning_path} />
          <TechStack stack={result.tech_stack} />

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="pt-4 text-center">
            <Button variant="secondary" onClick={reset}>
              Research Another Idea
            </Button>
          </motion.div>
        </div>

        <RepoChatPanel result={result} />
      </div>
    </div>
  );
}
