import { useEffect, useState } from 'react';

// Text that has finished typing once is shown instantly afterwards, so tab
// switches and reopening chat don't replay the same reveal.
const typedOnce = new Set<string>();

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

interface TypewriterOptions {
  msPerChar?: number;
  maxMs?: number;
  delayMs?: number;
}

export function useTypewriter(text: string, { msPerChar = 12, maxMs = 1800, delayMs = 0 }: TypewriterOptions = {}) {
  const skip = !text || typedOnce.has(text) || prefersReducedMotion();
  // Progress is tagged with the text it belongs to, so a new text starts at 0
  // on its first render instead of flashing the previous text's count.
  const [progress, setProgress] = useState({ text, count: 0 });

  useEffect(() => {
    if (skip) return;
    const duration = Math.min(maxMs, Math.max(250, text.length * msPerChar));
    // Wall-clock driven timer rather than rAF: rAF stops entirely in hidden or
    // occluded windows, which would leave the text invisible indefinitely.
    const start = performance.now() + delayMs;
    let timer = 0;
    const tick = () => {
      const next = Math.min(text.length, Math.floor((Math.max(0, performance.now() - start) / duration) * text.length));
      if (next >= text.length) typedOnce.add(text);
      setProgress({ text, count: next });
      if (next < text.length) timer = window.setTimeout(tick, 16);
    };
    timer = window.setTimeout(tick, 16);
    return () => window.clearTimeout(timer);
  }, [text, skip, msPerChar, maxMs, delayMs]);

  const count = skip ? text.length : progress.text === text ? progress.count : 0;
  return { shown: text.slice(0, count), done: count >= text.length };
}
