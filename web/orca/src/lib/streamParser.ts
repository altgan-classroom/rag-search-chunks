import type { StreamChunk } from "./types";

/**
 * Incremental parser for the chat stream.
 *
 * Understands, in any mix:
 *   - concatenated JSON objects   {..}{..}      (template default)
 *   - NDJSON                      {..}\n{..}\n
 *   - Server-Sent Events          data: {..}\n\n  (and `data: [DONE]`)
 *
 * The scanner is string-aware: braces and escaped quotes inside JSON strings
 * never affect object boundaries. Incomplete trailing data is kept until the
 * next push(). Malformed objects are skipped, never thrown.
 */
export class StreamParser {
  private buf = "";
  private pos = 0; // scan position inside buf
  private depth = 0;
  private start = -1;
  private inString = false;
  private escaped = false;

  push(text: string): StreamChunk[] {
    this.buf += text;
    const out: StreamChunk[] = [];
    const b = this.buf;

    for (; this.pos < b.length; this.pos++) {
      const ch = b[this.pos];

      if (this.depth === 0) {
        // Outside any object: only a '{' starts something. Everything else
        // (whitespace, `data:` prefixes, `[DONE]`, comments) is framing noise.
        if (ch === "{") {
          this.depth = 1;
          this.start = this.pos;
          this.inString = false;
          this.escaped = false;
        }
        continue;
      }

      if (this.inString) {
        if (this.escaped) this.escaped = false;
        else if (ch === "\\") this.escaped = true;
        else if (ch === '"') this.inString = false;
        continue;
      }

      if (ch === '"') this.inString = true;
      else if (ch === "{") this.depth++;
      else if (ch === "}") {
        this.depth--;
        if (this.depth === 0) {
          const chunk = parseChunk(b.slice(this.start, this.pos + 1));
          if (chunk) out.push(chunk);
          this.start = -1;
        }
      }
    }

    // Drop consumed input so the buffer stays small on long streams.
    if (this.depth === 0) {
      this.buf = "";
      this.pos = 0;
    } else if (this.start > 0) {
      this.buf = b.slice(this.start);
      this.pos -= this.start;
      this.start = 0;
    }
    return out;
  }

  /** Call when the stream ends; reports whether a truncated object was dropped. */
  end(): { truncated: boolean } {
    const truncated = this.depth > 0;
    this.buf = "";
    this.pos = 0;
    this.depth = 0;
    this.start = -1;
    return { truncated };
  }
}

function parseChunk(raw: string): StreamChunk | null {
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === "object" && typeof v.type === "string") {
      return { ...v, text: typeof v.text === "string" ? v.text : v.text == null ? "" : String(v.text) };
    }
  } catch (err) {
    console.warn("[stream] skipped malformed chunk", err);
  }
  return null;
}
