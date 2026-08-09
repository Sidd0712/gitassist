import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { useAppStore } from '../store/useAppStore';

interface ArchitectureDiagramProps {
  chart: string;
}

function readMermaidTheme() {
  const styles = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => styles.getPropertyValue(name).trim() || fallback;
  return {
    background: read('--color-bg', '#f2f2f3'),
    surface: read('--color-surface', '#e9e9ea'),
    text: read('--color-text', '#1d1f20'),
    accent: read('--color-accent', '#5980a6'),
    line: read('--color-neutral-500', '#98989b'),
  };
}

function renderDiagram(container: HTMLDivElement, chart: string) {
  let cleanChart = chart.trim();
  if (cleanChart.startsWith('```')) {
    cleanChart = cleanChart.replace(/^```(?:mermaid)?\n?/, '').replace(/\n?```$/, '');
  }

  const colors = readMermaidTheme();
  mermaid.initialize({
    startOnLoad: false,
    theme: 'base',
    themeVariables: {
      primaryColor: colors.surface,
      primaryTextColor: colors.text,
      primaryBorderColor: colors.accent,
      lineColor: colors.line,
      secondaryColor: colors.surface,
      tertiaryColor: colors.surface,
      background: colors.background,
      mainBkg: colors.surface,
      nodeBorder: colors.accent,
      clusterBkg: colors.surface,
      clusterBorder: colors.line,
      titleColor: colors.text,
      edgeLabelBackground: colors.background,
      fontFamily: 'Barlow, system-ui, sans-serif',
    },
  });

  const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return mermaid
    .render(id, cleanChart)
    .then(({ svg }) => {
      container.innerHTML = svg;
    })
    .catch((err) => {
      console.warn('Mermaid render error:', err);
      container.innerHTML = `<pre style="font-family:var(--font-mono);font-size:12px;white-space:pre-wrap;">${chart}</pre>`;
    });
}

function DiagramCanvas({ chart, minHeight }: { chart: string; minHeight: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const theme = useAppStore((s) => s.theme);

  useEffect(() => {
    if (!chart || !containerRef.current) return;
    renderDiagram(containerRef.current, chart);
  }, [chart, theme]);

  return (
    <div
      ref={containerRef}
      style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight, overflow: 'auto' }}
      className="diagram-canvas"
    />
  );
}

export function ArchitectureDiagram({ chart }: ArchitectureDiagramProps) {
  const [fullscreen, setFullscreen] = useState(false);

  if (!chart) return null;

  return (
    <>
      <div className="card blueprint elev-sm" style={{ padding: 14 }}>
        <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <div className="card-kicker">architecture_diagram · mermaid</div>
          <button className="btn btn-secondary btn-icon" onClick={() => setFullscreen(true)} aria-label="Expand">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
            </svg>
          </button>
        </div>
        <DiagramCanvas chart={chart} minHeight={220} />
      </div>

      {fullscreen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'color-mix(in srgb, var(--color-neutral-900) 60%, transparent)',
            zIndex: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 40,
          }}
          onClick={() => setFullscreen(false)}
        >
          <div
            className="card blueprint elev-lg"
            style={{ width: '100%', maxWidth: 900, background: 'var(--color-bg)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <h4 style={{ margin: 0 }}>Architecture diagram</h4>
              <button className="btn btn-icon btn-secondary" onClick={() => setFullscreen(false)} aria-label="Close">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
            <DiagramCanvas chart={chart} minHeight={460} />
          </div>
        </div>
      )}
    </>
  );
}
