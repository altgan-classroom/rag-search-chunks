import { describe, expect, it } from "vitest";
import { StreamParser } from "./streamParser";

const obj = (type: string, text: string) => JSON.stringify({ response_id: "1", type, text });
const texts = (cs: { text: string }[]) => cs.map((c) => c.text);

describe("StreamParser", () => {
  it("parses concatenated objects", () => {
    const p = new StreamParser();
    expect(texts(p.push(obj("tokens", "a") + obj("tokens", "b")))).toEqual(["a", "b"]);
  });

  it("handles an object split across pushes at every position", () => {
    const whole = obj("tokens", 'he said "hi" {x}') + obj("tokens", "!");
    for (let i = 1; i < whole.length; i++) {
      const p = new StreamParser();
      const out = [...p.push(whole.slice(0, i)), ...p.push(whole.slice(i))];
      expect(texts(out)).toEqual(['he said "hi" {x}', "!"]);
    }
  });

  it("ignores braces and escaped quotes inside strings", () => {
    const p = new StreamParser();
    const out = p.push(obj("tokens", '}{ \\ " {{{'));
    expect(texts(out)).toEqual(['}{ \\ " {{{']);
  });

  it("parses a context payload that is itself JSON text", () => {
    const ctx = JSON.stringify({ 0: { doc_name: "a", page_num: 3 } });
    const out = new StreamParser().push(obj("context", ctx));
    expect(out[0].type).toBe("context");
    expect(JSON.parse(out[0].text)[0].page_num).toBe(3);
  });

  it("supports NDJSON", () => {
    const out = new StreamParser().push(obj("tokens", "a") + "\n" + obj("tokens", "b") + "\n");
    expect(texts(out)).toEqual(["a", "b"]);
  });

  it("supports SSE framing, [DONE], comments and event lines", () => {
    const sse = `: ping\nevent: message\ndata: ${obj("tokens", "a")}\n\ndata: ${obj("tokens", "b")}\n\ndata: [DONE]\n\n`;
    expect(texts(new StreamParser().push(sse))).toEqual(["a", "b"]);
  });

  it("skips malformed objects and keeps going", () => {
    const out = new StreamParser().push('{"type": oops}' + obj("tokens", "ok"));
    expect(texts(out)).toEqual(["ok"]);
  });

  it("skips JSON without a string type", () => {
    expect(new StreamParser().push('{"foo":1}')).toEqual([]);
  });

  it("emits nothing for a partial object and reports truncation on end()", () => {
    const p = new StreamParser();
    expect(p.push('{"type":"tokens","te')).toEqual([]);
    expect(p.end().truncated).toBe(true);
  });

  it("stays correct over a long stream (buffer is trimmed)", () => {
    const p = new StreamParser();
    let n = 0;
    for (let i = 0; i < 2000; i++) n += p.push(obj("tokens", `t${i}`)).length;
    expect(n).toBe(2000);
  });

  it("preserves multi-byte text", () => {
    expect(texts(new StreamParser().push(obj("tokens", "héllo 日本 🙂")))).toEqual(["héllo 日本 🙂"]);
  });
});
