import { motion } from 'framer-motion';

interface InputFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  multiline?: boolean;
  rows?: number;
  id?: string;
}

export function InputField({
  value,
  onChange,
  placeholder = '',
  label,
  multiline = false,
  rows = 4,
  id,
}: InputFieldProps) {
  const baseClass =
    'w-full bg-white/[0.03] border border-white/10 rounded-xl px-5 py-4 text-text-primary placeholder-text-muted ' +
    'focus:outline-none focus:border-accent-violet/50 focus:shadow-[0_0_20px_rgba(139,92,246,0.15)] ' +
    'transition-all duration-300 font-[Inter] text-base resize-none';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full"
    >
      {label && (
        <label htmlFor={id} className="block text-sm font-medium text-text-secondary mb-2">
          {label}
        </label>
      )}
      {multiline ? (
        <textarea
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={rows}
          className={baseClass}
        />
      ) : (
        <input
          id={id}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={baseClass}
        />
      )}
    </motion.div>
  );
}
