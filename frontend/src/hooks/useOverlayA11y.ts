import { useEffect, type RefObject } from 'react';

const FOCUSABLE = 'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Escape-to-close for any overlay, plus an optional Tab focus trap for true
 * blocking modals (those with a dimmed backdrop over the whole page).
 */
export function useOverlayA11y(
  containerRef: RefObject<HTMLElement | null>,
  onClose: () => void,
  options: { active: boolean; trapFocus?: boolean },
) {
  const { active, trapFocus = false } = options;

  useEffect(() => {
    if (!active) return;

    if (trapFocus) {
      const previouslyFocused = document.activeElement as HTMLElement | null;
      const container = containerRef.current;
      const focusable = container?.querySelectorAll<HTMLElement>(FOCUSABLE);
      (focusable?.[0] ?? container)?.focus();
      return () => previouslyFocused?.focus?.();
    }
  }, [active, trapFocus, containerRef]);

  useEffect(() => {
    if (!active) return;

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
        return;
      }
      if (!trapFocus || e.key !== 'Tab') return;

      const container = containerRef.current;
      const focusable = container?.querySelectorAll<HTMLElement>(FOCUSABLE);
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [active, trapFocus, onClose, containerRef]);
}
