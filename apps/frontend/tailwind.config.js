/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Deep slate backgrounds
        slate: {
          950: '#020617',
          900: '#0f172a',
          850: '#111827',
          800: '#1e293b',
          750: '#233044',
          700: '#334155',
          600: '#475569',
          500: '#64748b',
          400: '#94a3b8',
          300: '#cbd5e1',
          200: '#e2e8f0',
          100: '#f1f5f9',
        },
        // Teal/Cyan AI accent
        cyan: {
          950: '#083344',
          900: '#0c4a6e',
          800: '#075985',
          700: '#0369a1',
          600: '#0284c7',
          500: '#06b6d4',
          400: '#22d3ee',
          300: '#67e8f9',
          200: '#a5f3fc',
          100: '#cffafe',
        },
        // Medical-specific semantic colors
        medical: {
          bg: '#0f172a',
          surface: '#1e293b',
          'surface-2': '#233044',
          border: '#334155',
          'border-subtle': '#1e293b',
          text: '#f8fafc',
          'text-muted': '#94a3b8',
          'text-dim': '#64748b',
          accent: '#06b6d4',
          'accent-bright': '#22d3ee',
          success: '#10b981',
          'success-dim': '#064e3b',
          warning: '#f59e0b',
          'warning-dim': '#451a03',
          error: '#ef4444',
          'error-dim': '#450a0a',
          'ai-glow': 'rgba(6, 182, 212, 0.15)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
        xs: ['0.75rem', { lineHeight: '1rem' }],
        sm: ['0.8125rem', { lineHeight: '1.25rem' }],
        base: ['0.875rem', { lineHeight: '1.375rem' }],
        lg: ['1rem', { lineHeight: '1.5rem' }],
        xl: ['1.125rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.25rem', { lineHeight: '1.875rem' }],
      },
      spacing: {
        sidebar: '240px',
        topbar: '52px',
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(6, 182, 212, 0.3)',
        'glow-cyan-sm': '0 0 10px rgba(6, 182, 212, 0.2)',
        'glow-green': '0 0 15px rgba(16, 185, 129, 0.25)',
        'glow-red': '0 0 15px rgba(239, 68, 68, 0.25)',
        'inset-top': 'inset 0 1px 0 rgba(255,255,255,0.06)',
        panel: '0 4px 24px rgba(0,0,0,0.4)',
      },
      backgroundImage: {
        'grid-subtle': 'radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px)',
        'gradient-medical': 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      },
      backgroundSize: {
        'grid-24': '24px 24px',
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scan': 'scan 2s ease-in-out infinite',
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'slide-in-up': 'slideInUp 0.2s ease-out',
      },
      keyframes: {
        scan: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '1' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideInRight: {
          from: { transform: 'translateX(16px)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        slideInUp: {
          from: { transform: 'translateY(8px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
