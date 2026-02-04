/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#050509',      // Deepest black-blue
        panel: '#0f1219',           // Secondary panel color
        primary: '#2563eb',         // The bright "AI Blue"
        secondary: '#1e293b',
        accent: '#818cf8',          // Purple accent
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}