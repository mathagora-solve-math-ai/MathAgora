/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        serif: ["'Instrument Serif'", "serif"],
      },
      boxShadow: {
        soft: "0 12px 40px -24px rgba(15, 23, 42, 0.35)",
      },
    },
  },
  plugins: [],
};
