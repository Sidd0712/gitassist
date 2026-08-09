import { useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { RepoList } from './RepoList';
import { LearningPath } from './LearningPath';
import { TechStack } from './TechStack';
import { ArchitectureDiagram } from './ArchitectureDiagram';
import { CitationPopover, type CitationDetail } from './CitationPopover';
import { ChatPanel } from './ChatPanel';

type TabId = 'repos' | 'learning' | 'architecture' | 'stack';

export function ResultsView() {
  const result = useAppStore((s) => s.result);
  const [activeTab, setActiveTab] = useState<TabId>('repos');
  const [activeCitation, setActiveCitation] = useState<CitationDetail | null>(null);

  if (!result) return null;

  if (result.status === 'error') {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 20px' }}>
        <div style={{ width: '100%', maxWidth: 480 }}>
          <h6 style={{ color: 'var(--color-accent)' }}>research failed</h6>
          <h3 style={{ marginBottom: 'var(--space-2)' }}>Something went wrong</h3>
          <p className="text-muted">{result.error || 'The research pipeline hit an error. Try again with a more specific idea.'}</p>
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
    <div style={{ flex: 1, padding: '28px 32px 100px', maxWidth: 1180, margin: '0 auto', width: '100%', boxSizing: 'border-box' }}>
      <h6 style={{ color: 'var(--color-accent)' }}>idea summary</h6>
      <p style={{ maxWidth: 760, fontSize: 15 }}>{result.idea_summary}</p>

      {result.error && (
        <p style={{ color: '#b5493b', fontSize: 13 }} role="alert">
          {result.error}
        </p>
      )}

      <div className="seg" style={{ margin: 'var(--space-3) 0 var(--space-4)' }}>
        {tabs.map((t) => (
          <label key={t.id} className="seg-opt">
            <input type="radio" name="restab" checked={activeTab === t.id} onChange={() => setActiveTab(t.id)} />
            {t.label}
          </label>
        ))}
      </div>

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

      <CitationPopover
        citation={activeCitation}
        repoUrl={activeCitation ? repoUrlByName.get(activeCitation.repo_full_name) : undefined}
        onClose={() => setActiveCitation(null)}
      />

      <ChatPanel />
    </div>
  );
}
