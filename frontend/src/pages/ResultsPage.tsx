import { motion } from 'framer-motion';
import { Button } from '../components/Button';
import { GlassCard } from '../components/GlassCard';
import { MermaidDiagram } from '../components/MermaidDiagram';
import { useAppStore } from '../store/useAppStore';
import type { AnalysisResponse } from '../types';

function IntentPanel({ result }: { result: AnalysisResponse }) {
  const chips = [
    ...result.keywords.capabilities,
    ...result.keywords.frameworks,
    ...result.keywords.languages,
  ].slice(0, 12);

  return (
    <GlassCard glow="fuchsia" className="space-y-4">
      <div>
        <h3 className="text-sm text-text-muted uppercase tracking-wider mb-2">Resolved Idea</h3>
        <p className="text-text-primary text-lg">{result.idea_summary}</p>
      </div>

      {chips.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {chips.map((chip) => (
            <span
              key={chip}
              className="px-3 py-1 rounded-full text-sm bg-accent-violet/15 text-accent-violet border border-accent-violet/20"
            >
              {chip}
            </span>
          ))}
        </div>
      )}

      {result.assumptions.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-text-primary mb-2">Assumptions</h4>
          <div className="flex flex-wrap gap-2">
            {result.assumptions.map((assumption) => (
              <span key={assumption} className="px-3 py-2 rounded-xl bg-white/[0.03] text-text-secondary text-sm">
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
      <h3 className="text-xl font-semibold font-[Outfit] mb-4">Useful Reference Repositories</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {repos.map((repo, index) => (
          <GlassCard key={repo.full_name} delay={0.08 * index} glow={index === 0 ? 'violet' : 'none'}>
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <a
                  href={repo.html_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-cyan hover:underline font-semibold text-base"
                >
                  {repo.full_name}
                </a>
                <p className="text-text-secondary text-sm mt-2">{repo.fit_summary || repo.description || 'Reference repo'}</p>
              </div>
              <div className="text-right shrink-0">
                <div className="text-yellow-400 text-sm">★ {repo.stars.toLocaleString()}</div>
                {repo.fit_score !== undefined && (
                  <div className="text-xs text-text-muted mt-1">fit {(repo.fit_score * 100).toFixed(0)}%</div>
                )}
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-3">
              {repo.reference_type && (
                <span className="px-2.5 py-1 rounded-full text-xs uppercase tracking-wider bg-accent-fuchsia/15 text-accent-fuchsia">
                  {repo.reference_type.replace('_', ' ')}
                </span>
              )}
              {repo.language && (
                <span className="px-2.5 py-1 rounded-full text-xs bg-white/5 text-text-muted">{repo.language}</span>
              )}
              {repo.covered_primary?.map((capability) => (
                <span key={capability} className="px-2.5 py-1 rounded-full text-xs bg-accent-cyan/10 text-accent-cyan">
                  covers {capability}
                </span>
              ))}
            </div>

            {descriptions[index] && <p className="text-sm text-text-primary leading-relaxed">{descriptions[index]}</p>}
            {repo.missing_primary && repo.missing_primary.length > 0 && (
              <p className="text-xs text-text-muted mt-3">Missing: {repo.missing_primary.join(', ')}</p>
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
      <h3 className="text-xl font-semibold font-[Outfit] mb-5">Build Path</h3>
      <div className="space-y-5">
        {steps.map((step) => (
          <div key={step.step_number} className="flex gap-4">
            <div className="w-10 h-10 rounded-full bg-gradient-to-r from-accent-violet to-accent-fuchsia flex items-center justify-center text-white font-semibold shrink-0">
              {step.step_number}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs uppercase tracking-wider text-text-muted">{step.milestone}</p>
              <h4 className="font-semibold text-text-primary mt-1">{step.title}</h4>
              <p className="text-text-secondary text-sm mt-2">{step.description}</p>
              {step.concepts.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {step.concepts.map((concept) => (
                    <span key={concept} className="px-2.5 py-1 rounded-full text-xs bg-white/5 text-text-muted">
                      {concept}
                    </span>
                  ))}
                </div>
              )}
              {step.resources.length > 0 && (
                <div className="flex flex-wrap gap-3 mt-3">
                  {step.resources.map((resource) => (
                    <a
                      key={resource}
                      href={resource.startsWith('http') ? resource : '#'}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-accent-cyan hover:underline"
                    >
                      {resource.length > 54 ? resource.slice(0, 54) + '...' : resource}
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
      <h3 className="text-xl font-semibold font-[Outfit] mb-4">Recommended Tech Stack</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {stack.map((tech) => (
          <div key={tech.name} className="glass p-4 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-semibold text-text-primary">{tech.name}</span>
              <span className="px-2 py-0.5 rounded text-[10px] bg-accent-violet/20 text-accent-violet uppercase tracking-wider">
                {tech.category}
              </span>
            </div>
            <p className="text-sm text-text-secondary">{tech.why_recommended}</p>
            {tech.supported_by.length > 0 && (
              <p className="text-xs text-text-muted mt-2">Supported by: {tech.supported_by.join(', ')}</p>
            )}
            {tech.pros.length > 0 && (
              <div className="text-xs text-green-300/85 mt-3">{tech.pros.join(' · ')}</div>
            )}
            {tech.cons.length > 0 && (
              <div className="text-xs text-red-300/70 mt-2">{tech.cons.join(' · ')}</div>
            )}
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

export function ResultsPage() {
  const { result, reset, error } = useAppStore();

  if (!result) return null;

  return (
    <div className="relative z-10 min-h-screen px-4 py-12 max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-8"
      >
        <Button variant="ghost" size="sm" onClick={reset}>
          Back
        </Button>
        <h2 className="text-2xl font-bold font-[Outfit] gradient-text">Build Recommendations</h2>
        <div className="w-16" />
      </motion.div>

      {error && (
        <div className="glass border-red-500/30 bg-red-500/10 p-4 rounded-xl text-red-300 text-sm mb-6">{error}</div>
      )}

      <div className="space-y-8">
        <IntentPanel result={result} />
        <RepoCards repos={result.repositories} descriptions={result.repo_descriptions} />
        <MermaidDiagram chart={result.architecture_diagram} />
        <BuildPath steps={result.learning_path} />
        <TechStack stack={result.tech_stack} />
      </div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="text-center mt-12">
        <Button variant="secondary" onClick={reset}>
          Research Another Idea
        </Button>
      </motion.div>
    </div>
  );
}
