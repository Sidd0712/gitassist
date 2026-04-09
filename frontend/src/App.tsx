import { LoadingSpinner } from './components/LoadingSpinner';
import { ClarificationPage } from './pages/ClarificationPage';
import { HomePage } from './pages/HomePage';
import { ResultsPage } from './pages/ResultsPage';
import { useAppStore } from './store/useAppStore';

function App() {
  const view = useAppStore((state) => state.view);

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      {view === 'home' && <HomePage />}
      {view === 'loading' && (
        <div className="relative z-10 px-4 py-16">
          <LoadingSpinner />
        </div>
      )}
      {view === 'clarify' && <ClarificationPage />}
      {view === 'results' && <ResultsPage />}
    </div>
  );
}

export default App;
