const graphNodes = [
  "router",
  "proactive_check",
  "dose_rag",
  "dose_tavily",
  "dose_extractor",
  "calculator",
  "rag_retriever",
  "vlm_analyzer",
  "source_vetter",
  "synthesizer",
];

const tools = [
  {
    name: "Tool A",
    title: "Deterministic Calculator",
    text: "Computes concentration, draw volume, U100/U40 units, and recommended BAC water for convenient dosing.",
  },
  {
    name: "Tool B",
    title: "Pinecone PubMed RAG",
    text: "Retrieves abstract chunks and PMIDs for mechanisms, dosing context, and citation support.",
  },
  {
    name: "Tool C",
    title: "Bloodwork VLM",
    text: "Reads uploaded lab panels and flags relevant biomarkers for growth-factor peptide safety context.",
  },
  {
    name: "Tool D",
    title: "Tavily Web Search",
    text: "Finds web/community dosing protocols and vendor/source reputation signals.",
  },
];

const modes = [
  {
    title: "GPT-only baselines",
    text: "Zero-shot, few-shot, and CoT prompts return the same four-part JSON structure without tools.",
  },
  {
    title: "Full agent",
    text: "LangGraph routes to dose research, calculator, RAG, VLM, source vetting, and final synthesis.",
  },
  {
    title: "Ablations",
    text: "No-reasoning, no-calculator, and RAG-only variants isolate which system pieces improve metrics.",
  },
];

export default function Architecture() {
  return (
    <div className="space-y-10">
      <section className="panel p-6 sm:p-8">
        <p className="eyebrow">System architecture</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">LangGraph agent with deterministic tools</h1>
        <p className="mt-4 max-w-3xl text-slate-300">
          Full-agent mode turns one query into a tool-aware workflow: route intent, research missing
          dose information, calculate reconstitution, retrieve PubMed evidence, inspect optional
          bloodwork, vet sources, and synthesize a structured answer.
        </p>
      </section>

      <section className="panel p-6">
        <h2 className="text-2xl font-semibold text-white">LangGraph Flow</h2>
        <div className="mt-6 flex flex-wrap items-center gap-3">
          {graphNodes.map((node, index) => (
            <div key={node} className="flex items-center gap-3">
              <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 px-4 py-3 text-sm font-semibold text-cyan-100">
                {node}
              </div>
              {index < graphNodes.length - 1 && <span className="text-slate-500">→</span>}
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-semibold text-white">Tool Layer</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {tools.map((tool) => (
            <article key={tool.name} className="panel-soft p-5">
              <span className="metric-pill">{tool.name}</span>
              <h3 className="mt-4 text-lg font-semibold text-white">{tool.title}</h3>
              <p className="mt-3 text-sm leading-6 text-slate-400">{tool.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        {modes.map((mode) => (
          <article key={mode.title} className="panel p-6">
            <h3 className="text-xl font-semibold text-white">{mode.title}</h3>
            <p className="mt-3 text-sm leading-6 text-slate-400">{mode.text}</p>
          </article>
        ))}
      </section>
    </div>
  );
}
