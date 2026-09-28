import type { ReactNode } from 'react';
import { useTypewriter } from '../hooks/useTypewriter';

interface TypewriterProps {
  text: string;
  delayMs?: number;
  msPerChar?: number;
  maxMs?: number;
  /** Rendered after the text (e.g. citation markers); revealed once typing ends. */
  after?: ReactNode;
}

/**
 * Types `text` out with a block cursor. The full text is laid out underneath
 * (transparent, still read by screen readers) so the block reserves its final
 * size up front and nothing below it jumps while characters appear.
 */
export function Typewriter({ text, delayMs, msPerChar, maxMs, after }: TypewriterProps) {
  const { shown, done } = useTypewriter(text, { delayMs, msPerChar, maxMs });

  if (done) {
    return (
      <>
        {text}
        {after}
      </>
    );
  }

  return (
    <span className="typewriter">
      <span className="typewriter-ghost">
        {text}
        {after}
      </span>
      <span className="typewriter-live" aria-hidden="true">
        {shown}
        <span className="type-cursor" />
      </span>
    </span>
  );
}
