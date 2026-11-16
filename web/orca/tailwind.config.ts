import type { Config } from "tailwindcss";

const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: v("bg"),
        surface: v("surface"),
        surface2: v("surface-2"),
        line: v("line"),
        fg: v("fg"),
        muted: v("muted"),
        brand: { DEFAULT: v("brand"), soft: v("brand-soft"), violet: v("violet"), cyan: v("cyan") },
        danger: v("danger"),
        success: v("success"),
      },
      fontFamily: {
        sans: ['"Inter"', "ui-sans-serif", "system-ui", "-apple-system", '"Segoe UI"', "Roboto", "sans-serif"],
        mono: ["ui-monospace", '"SF Mono"', "Menlo", "Consolas", "monospace"],
      },
      backgroundImage: {
        "brand-gradient": "linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 120%)",
        "brand-text": "linear-gradient(90deg, #6366f1, #a855f7, #06b6d4)",
      },
      boxShadow: {
        card: "0 1px 2px rgb(var(--shadow) / .06), 0 4px 16px rgb(var(--shadow) / .06)",
        pop: "0 8px 30px rgb(var(--shadow) / .18)",
        glow: "0 6px 24px rgb(99 102 241 / .35)",
      },
      keyframes: {
        rise: { from: { opacity: "0", transform: "translateY(8px)" }, to: { opacity: "1", transform: "none" } },
        caret: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        blob: { "0%,100%": { transform: "translate(0,0) scale(1)" }, "50%": { transform: "translate(20px,-16px) scale(1.08)" } },
      },
      animation: {
        rise: "rise .25s ease-out both",
        caret: "caret 1s steps(1) infinite",
        shimmer: "shimmer 2s linear infinite",
        blob: "blob 12s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
