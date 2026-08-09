import { Nav } from './components/Nav';
import { IdeaForm } from './components/IdeaForm';
import { ProgressList } from './components/ProgressList';
import { ClarificationDialog } from './components/ClarificationDialog';
import { ResultsView } from './components/ResultsView';
import { useAppStore } from './store/useAppStore';

function App() {
  const view = useAppStore((s) => s.view);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)', display: 'flex', flexDirection: 'column' }}>
      <Nav />

      {view === 'home' && <IdeaForm />}
      {view === 'loading' && <ProgressList />}
      {view === 'clarify' && (
        <>
          <IdeaForm />
          <ClarificationDialog />
        </>
      )}
      {view === 'results' && <ResultsView />}
    </div>
  );
}

export default App;
