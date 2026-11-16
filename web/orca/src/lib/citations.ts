import { DOCS_BASE_URL } from "./config";

export interface Source {
  index: number;
  docName: string;
  page: string;
  url: string;
  chunk?: string;
}

/** Parses the `context` chunk payload. Never throws. */
export function parseSources(raw: string | undefined): Source[] {
  if (!raw || raw === "{}") return [];
  try {
    const obj = JSON.parse(raw.replace(/\n/g, " "));
    if (!obj || typeof obj !== "object") return [];
    return Object.values<Record<string, unknown>>(obj as Record<string, Record<string, unknown>>)
      .filter((v) => v && typeof v === "object" && v.doc_name)
      .map((v, i) => {
        const docName = String(v.doc_name).replace(/\.md$/, "");
        const page = v.page_num == null ? "" : String(v.page_num);
        return {
          index: i + 1,
          docName,
          page,
          url: `${DOCS_BASE_URL}/${docName}.pdf${page ? `#page=${page}` : ""}`,
          chunk: typeof v.chunk === "string" ? v.chunk : undefined,
        };
      });
  } catch {
    return [];
  }
}
