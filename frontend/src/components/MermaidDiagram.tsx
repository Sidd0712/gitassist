import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#8b5cf6',
    primaryTextColor: '#f1f0f5',
    primaryBorderColor: '#6d28d9',
    lineColor: '#a78bfa',
    secondaryColor: '#d946ef',
    tertiaryColor: '#0f0a1a',
    background: '#0a0612',
    mainBkg: '#1a1030',
    nodeBorder: '#8b5cf6',
    clusterBkg: '#1a1030',
    titleColor: '#f1f0f5',
    edgeLabelBackground: '#1a1030',
  },
  fontFamily: 'Inter, sans-serif',
});

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

export function MermaidDiagram({ chart, className = '' }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chart || !containerRef.current) return;

    const render = async () => {
      try {
        let cleanChart = chart.trim();
        if (cleanChart.startsWith('```')) {
          cleanChart = cleanChart.replace(/^```(?:mermaid)?\n?/, '').replace(/\n?```$/, '');
        }

        const id = `mermaid-${Date.now()}`;
        const { svg } = await mermaid.render(id, cleanChart);
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch (err) {
        console.warn('Mermaid render error:', err);
        if (containerRef.current) {
          containerRef.current.innerHTML = `<pre class="text-text-muted text-sm p-4">${chart}</pre>`;
        }
      }
    };

    render();
  }, [chart]);

  if (!chart) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`glass p-6 overflow-x-auto ${className}`}
    >
      <h3 className="text-lg font-semibold text-text-primary mb-4 font-[Outfit]">Recommended Architecture</h3>
      <div ref={containerRef} className="flex justify-center [&>svg]:max-w-full" />
    </motion.div>
  );
}
