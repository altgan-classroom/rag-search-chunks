const env = import.meta.env as Record<string, string | undefined>;

const trim = (s: string) => s.replace(/\/+$/, "");

/** Retrieval API base URL (build-time, set VITE_API_URL). */
export const API_URL = trim(env.VITE_API_URL || "http://localhost:8080");

/** Optional API key sent as X-API-Key (build-time, VITE_API_KEY). Dev convenience only: a key baked into a public
 *  bundle is not a secret. In production put the UI behind your SSO/gateway and inject credentials server-side. */
export const API_KEY = env.VITE_API_KEY || "";

/** Where cited PDFs are hosted (build-time, set VITE_DOCS_BASE_URL). */
export const DOCS_BASE_URL = trim(
  env.VITE_DOCS_BASE_URL || "https://rag-prod-ingested-docs.s3.ap-south-1.amazonaws.com",
);
