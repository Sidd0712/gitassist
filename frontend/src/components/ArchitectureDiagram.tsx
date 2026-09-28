import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { useAppStore } from '../store/useAppStore';
import { useOverlayA11y } from '../hooks/useOverlayA11y';

interface ArchitectureDiagramProps {
  chart: string;
}

function readMermaidTheme() {
  const styles = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => styles.getPropertyValue(name).trim() || fallback;
  return {
    background: read('--color-bg', '#0b0c0e'),
    surface: read('--color-bg-input', '#101215'),
    text: read('--color-text', '#d9dce1'),
    accent: read('--color-accent', '#4fd671'),
    line: read('--color-divider-strong', '#33373d'),
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
      fontFamily: 'ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace',
    },
  });

  const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return mermaid
    .render(id, cleanChart)
    .then(({ svg }) => {
      container.innerHTML = svg;
      // Mermaid stamps its own computed max-width as an inline style, which
      // beats our CSS max-width:100% by specificity and lets wide/tall
      // diagrams overflow their container uncropped. Override it directly.
      const svgEl = container.querySelector('svg');
      if (svgEl) {
        svgEl.style.maxWidth = '100%';
        svgEl.style.height = 'auto';
        svgEl.style.display = 'block';
      }
    })
    .catch((err) => {
      console.warn('Mermaid render error:', err);
      container.innerHTML = `<pre style="font-family:var(--font-mono);font-size:12px;white-space:pre-wrap;">${chart}</pre>`;
    });
}

function DiagramCanvas({ chart, minHeight, maxHeight }: { chart: string; minHeight: number; maxHeight: number | string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const theme = useAppStore((s) => s.theme);

  useEffect(() => {
    if (!chart || !containerRef.current) return;
    renderDiagram(containerRef.current, chart);
  }, [chart, theme]);

  return (
    <div
      ref={containerRef}
      style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight, maxHeight, overflow: 'auto' }}
      className="diagram-canvas"
    />
  );
}

export function ArchitectureDiagram({ chart }: ArchitectureDiagramProps) {
  const [fullscreen, setFullscreen] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);
  useOverlayA11y(modalRef, () => setFullscreen(false), { active: fullscreen, trapFocus: true });

  if (!chart) return null;

  return (
    <>
      <div className="card" style={{ padding: 0 }}>
        <div className="file-head">
          <span className="fp">+++ b/architecture.mmd</span>
          <button className="btn btn-secondary btn-icon" onClick={() => setFullscreen(true)} aria-label="Expand" style={{ marginLeft: 'auto' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3" />
            </svg>
          </button>
        </div>
        <div style={{ padding: 14 }}>
          <DiagramCanvas chart={chart} minHeight={220} maxHeight={480} />
        </div>
      </div>

      {fullscreen && (
        <div
          className="ga-fade-in"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'color-mix(in srgb, black 65%, transparent)',
            zIndex: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 40,
          }}
          onClick={() => setFullscreen(false)}
        >
          <div
            ref={modalRef}
            className="card ga-pop-in"
            style={{
              width: '100%',
              maxWidth: 'min(1400px, calc(100vw - 80px))',
              maxHeight: 'calc(100vh - 80px)',
              overflowY: 'auto',
              background: 'var(--color-bg-raised)',
              padding: 0,
            }}
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="diagram-modal-title"
            tabIndex={-1}
          >
            <div className="file-head">
              <span className="fp" id="diagram-modal-title">+++ b/architecture.mmd</span>
              <button className="btn btn-icon btn-secondary" onClick={() => setFullscreen(false)} aria-label="Close" style={{ marginLeft: 'auto' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
            <div style={{ padding: 14 }}>
              <DiagramCanvas chart={chart} minHeight={460} maxHeight="calc(100vh - 220px)" />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
