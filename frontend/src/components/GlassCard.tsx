import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: 'violet' | 'fuchsia' | 'none';
  delay?: number;
}

export function GlassCard({ children, className = '', hover = true, glow = 'none', delay = 0 }: GlassCardProps) {
  const glowClass = glow === 'violet' ? 'glow-violet' : glow === 'fuchsia' ? 'glow-fuchsia' : '';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: 'easeOut' }}
      whileHover={hover ? { scale: 1.01, y: -2 } : undefined}
      className={`glass ${glowClass} p-6 transition-all duration-300 ${className}`}
    >
      {children}
    </motion.div>
  );
}
