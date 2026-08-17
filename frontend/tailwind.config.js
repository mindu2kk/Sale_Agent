/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        background: '#F5F7FB',
        surface: '#FFFFFF',
        'surface-muted': '#F8FAFC',
        'surface-card': '#FAFBFF',
        primary: '#2563EB',
        'primary-dark': '#1D4ED8',
        'primary-light': '#EFF6FF',
        text: {
          primary: '#0F172A',
          secondary: '#475569',
          muted: '#94A3B8',
          subtle: '#CBD5E1',
        },
        border: {
          DEFAULT: '#E2E8F0',
          soft: '#EEF2F8',
          strong: '#CBD5E1',
        },
        price: '#E11D48',
        'price-dark': '#C4132E',
        success: '#059669',
        'spec-bg': '#F1F5F9',
        'spec-text': '#475569',
      },
      borderRadius: {
        card: '20px',
        drawer: '24px',
        xl: '12px',
        '2xl': '16px',
        '3xl': '20px',
        '4xl': '24px',
      },
      boxShadow: {
        card: '0 2px 8px rgba(15, 23, 42, 0.05)',
        'card-hover': '0 12px 32px rgba(15, 23, 42, 0.10)',
        drawer: '-4px 0 32px rgba(15, 23, 42, 0.08)',
        nav: '0 1px 0 #E2E8F0',
        tray: '0 -8px 32px rgba(15, 23, 42, 0.10)',
        'ai-btn': '0 6px 24px rgba(37, 99, 235, 0.18)',
        modal: '0 24px 64px rgba(15, 23, 42, 0.18)',
      },
      fontSize: {
        'price-lg': ['22px', { fontWeight: '700', lineHeight: '1.1' }],
        'price-md': ['18px', { fontWeight: '700', lineHeight: '1.1' }],
        'price-sm': ['15px', { fontWeight: '700', lineHeight: '1.1' }],
      },
    },
  },
  plugins: [],
}