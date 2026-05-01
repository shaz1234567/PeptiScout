import { useMemo, useState } from "react";
import { postQuery } from "../lib/api.js";

const modes = [
  "baseline-zero-shot",
  "baseline-few-shot",
  "baseline-cot",
  "full-agent",
];

const suggestions = [
  "I have a 5mg vial of BPC-157 for tendon healing, how much BAC water should I use?",
  "What pathways does BPC-157 activate for tendon and tissue repair?",
  "What are the contraindications for PT-141?",
];

const sections = [
  ["protocol", "Protocol"],
  ["moa", "MOA"],
  ["good_bad", "The Good / The Bad"],
  ["audit_trail", "Audit Trail"],
];

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const value = String(reader.result || "");
      resolve(value.includes(",") ? value.split(",")[1] : value);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function Demo() {
  const [text, setText] = useState(suggestions[0]);
  const [mode, setMode] = useState("full-agent");
  const [image, setImage] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const modeDescription = useMemo(() => {
    if (mode === "full-agent") return "LangGraph agent with RAG, Tavily, calculator, VLM, and trace.";
    if (mode === "baseline-cot") return "GPT baseline with hidden chain-of-thought style planning prompt.";
    if (mode === "baseline-few-shot") return "GPT baseline with peptide examples in the prompt.";
    return "Direct structured GPT baseline with no tools.";
  }, [mode]);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResponse(null);
    try {
      const payload = {
        text,
        mode,
        image_base64: image ? await fileToBase64(image) : null,
        image_type: image?.type || null,
      };
      setResponse(await postQuery(payload));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[0.95fr_1.05fr]">
      <section className="panel p-6">
        <p className="eyebrow">Interactive demo</p>
        <h1 className="mt-3 text-3xl font-semibold text-white">Ask PeptiScout a peptide question</h1>
        <p className="mt-3 text-sm leading-6 text-slate-400">
          Choose a model mode, optionally attach a bloodwork image, and inspect the structured
          four-part response. Full-agent mode exposes the ReAct trace.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-5">
          <div>
            <label className="text-sm font-medium text-slate-200" htmlFor="question">
              Question
            </label>
            <textarea
              id="question"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={7}
              className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-950/80 p-4 text-sm text-white outline-none ring-cyan-300/30 transition placeholder:text-slate-600 focus:border-cyan-300/60 focus:ring-4"
              placeholder="Ask about dosing, MOA, safety, or vendor/source context..."
            />
          </div>

          <div>
            <label className="text-sm font-medium text-slate-200" htmlFor="mode">
              Mode
            </label>
            <select
              id="mode"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-950/80 p-3 text-white outline-none focus:border-cyan-300/60"
            >
              {modes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <p className="mt-2 text-xs text-slate-500">{modeDescription}</p>
          </div>

          <div>
            <label className="text-sm font-medium text-slate-200" htmlFor="image">
              Bloodwork image upload
            </label>
            <input
              id="image"
              type="file"
              accept="image/*"
              onChange={(e) => setImage(e.target.files?.[0] || null)}
              className="mt-2 block w-full rounded-2xl border border-dashed border-white/15 bg-white/[0.03] p-3 text-sm text-slate-300 file:mr-4 file:rounded-full file:border-0 file:bg-cyan-300 file:px-4 file:py-2 file:font-semibold file:text-slate-950"
            />
            <p className="mt-2 text-xs text-slate-500">
              Used by full-agent mode for VLM bloodwork analysis when an image is attached.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setText(s)}
                className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-300 transition hover:border-cyan-300/50 hover:text-cyan-100"
              >
                {s}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="w-full rounded-2xl bg-cyan-300 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? "Querying PeptiScout..." : "Run Query"}
          </button>
        </form>
      </section>

      <section className="space-y-4">
        {error && (
          <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-100">
            {error}
          </div>
        )}

        {!response && !error && (
          <div className="panel grid min-h-[420px] place-items-center p-8 text-center">
            <div>
              <p className="eyebrow">Awaiting query</p>
              <h2 className="mt-3 text-2xl font-semibold text-white">Structured output appears here</h2>
              <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">
                Responses are normalized into Protocol, MOA, Good/Bad, and Audit Trail cards.
              </p>
            </div>
          </div>
        )}

        {response && (
          <>
            <div className="grid gap-4">
              {sections.map(([key, label]) => (
                <article key={key} className="panel-soft p-5">
                  <h2 className="text-lg font-semibold text-cyan-100">{label}</h2>
                  <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-300">
                    {response[key] || "No content returned."}
                  </p>
                </article>
              ))}
            </div>

            {mode === "full-agent" && response.react_trace && (
              <details className="panel-soft p-5">
                <summary className="cursor-pointer text-lg font-semibold text-cyan-100">
                  ReAct Trace
                </summary>
                <pre className="mt-4 max-h-[520px] overflow-auto rounded-2xl bg-slate-950 p-4 text-xs leading-5 text-slate-300">
                  {response.react_trace}
                </pre>
              </details>
            )}
          </>
        )}
      </section>
    </div>
  );
}
