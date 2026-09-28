import { useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { RepoList } from './RepoList';
import { LearningPath } from './LearningPath';
import { TechStack } from './TechStack';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { CitationPopover, type CitationDetail } from './CitationPopover';
import { ChatPanel } from './ChatPanel';
import { Typewriter } from './Typewriter';

type TabId = 'repos' | 'learning' | 'architecture' | 'stack';

export function ResultsView() {
  const result = useAppStore((s) => s.result);
  const submitIdea = useAppStore((s) => s.submitIdea);
  const resetToHome = useAppStore((s) => s.reset);
  const [activeTab, setActiveTab] = useState<TabId>('repos');
  const [activeCitation, setActiveCitation] = useState<CitationDetail | null>(null);

  if (!result) return null;

  if (result.status === 'error') {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
        <div style={{ width: '100%', maxWidth: 480 }}>
          <h3 style={{ marginBottom: 'var(--space-2)' }}>Something went wrong</h3>
          <p className="text-muted">{result.error || 'The research pipeline hit an error. Try again with a more specific idea.'}</p>
          <div style={{ display: 'flex', gap: 8, marginTop: 'var(--space-3)' }}>
            <button className="btn btn-primary" onClick={submitIdea}>
              Try again
            </button>
            <button className="btn btn-secondary" onClick={resetToHome}>
              Start over
            </button>
          </div>
        </div>
      </div>
    );
  }

  const tabs: { id: TabId; label: string }[] = [
    { id: 'repos', label: `Repos (${result.repositories.length})` },
    { id: 'learning', label: 'Learning path' },
    { id: 'architecture', label: 'Architecture' },
    { id: 'stack', label: `Tech stack` },
  ];

  const repoDescriptionCitations = result.evidence?.repo_descriptions ?? [];
  const repoUrlByName = new Map(result.repositories.map((r) => [r.full_name, r.html_url]));

  return (
    <div className="results-page">
      <h1 className="results-title">
        <Typewriter text={result.idea_summary} msPerChar={18} maxMs={1400} />
      </h1>

      {result.error && (
        <p style={{ color: 'var(--color-danger)', fontSize: 13, marginTop: 'var(--space-3)' }} role="alert">
          {result.error}
        </p>
      )}

      <div className="seg" style={{ margin: '28px 0 24px' }}>
        {tabs.map((t) => (
          <label key={t.id} className="seg-opt">
            <input type="radio" name="restab" checked={activeTab === t.id} onChange={() => setActiveTab(t.id)} />
            {t.label}
          </label>
        ))}
      </div>

      <div key={activeTab} className="ga-fade-in">
        {activeTab === 'repos' && (
          <RepoList
            repositories={result.repositories}
            descriptions={result.repo_descriptions}
            citations={repoDescriptionCitations}
            onCite={(c) => setActiveCitation(c)}
          />
        )}
        {activeTab === 'learning' && <LearningPath steps={result.learning_path} />}
        {activeTab === 'architecture' && <ArchitectureDiagram chart={result.architecture_diagram} />}
        {activeTab === 'stack' && <TechStack items={result.tech_stack} />}
      </div>

      <CitationPopover
        citation={activeCitation}
        repoUrl={activeCitation ? repoUrlByName.get(activeCitation.repo_full_name) : undefined}
        onClose={() => setActiveCitation(null)}
      />

      <ChatPanel />
    </div>
  );
}
