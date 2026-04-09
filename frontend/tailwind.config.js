/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        heading: ['Outfit', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
      },
      colors: {
        'bg-primary': '#0a0612',
        'bg-card': 'rgba(255, 255, 255, 0.04)',
        'bg-card-hover': 'rgba(255, 255, 255, 0.08)',
        'border-glass': 'rgba(255, 255, 255, 0.08)',
        'border-glass-hover': 'rgba(255, 255, 255, 0.15)',
        'accent-violet': '#8b5cf6',
        'accent-fuchsia': '#d946ef',
        'accent-cyan': '#22d3ee',
        'text-primary': '#f1f0f5',
        'text-secondary': '#a8a3b5',
        'text-muted': '#6b6580',
      },
    },
  },
  plugins: [],
}
