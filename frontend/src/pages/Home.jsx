import { Link } from "react-router-dom";

const features = [
  {
    title: "Protocol",
    text: "Reconstitution math, dose context, and syringe-unit guidance in a structured format.",
  },
  {
    title: "MOA",
    text: "Pathway-level summaries grounded by PubMed-style retrieval and synthesis.",
  },
  {
    title: "Good / Bad",
    text: "Benefits, contraindications, cautions, and monitoring prompts in one section.",
  },
  {
    title: "Audit Trail",
    text: "PMID-first citations and source checks so claims can be inspected.",
  },
];

export default function Home() {
  return (
    <div className="space-y-14">
      <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
        <div className="space-y-7">
          <p className="eyebrow">Research peptide intelligence</p>
          <div className="space-y-5">
            <h1 className="max-w-4xl text-5xl font-semibold tracking-tight text-white sm:text-6xl">
              PeptiScout AI turns peptide questions into structured, auditable research briefs.
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-slate-300">
              Built for peptide researchers, students, and review workflows, PeptiScout combines
              baseline LLM responses, LangGraph tools, deterministic reconstitution math, PubMed
              RAG, Tavily search, and bloodwork image analysis.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              to="/demo"
              className="rounded-full bg-cyan-300 px-6 py-3 font-semibold text-slate-950 shadow-lg shadow-cyan-400/20 transition hover:bg-cyan-200"
            >
              Try the Demo
            </Link>
            <Link
              to="/results"
              className="rounded-full border border-white/15 px-6 py-3 font-semibold text-white transition hover:border-cyan-300/50 hover:bg-white/10"
            >
              View Results
            </Link>
          </div>
        </div>
        <div className="panel p-6">
          <div className="rounded-2xl border border-cyan-300/20 bg-slate-950/80 p-5">
            <p className="text-sm text-cyan-200">Example query</p>
            <p className="mt-3 text-xl font-medium text-white">
              “I have a 5mg vial of BPC-157 for tendon healing. How much BAC water should I use?”
            </p>
          </div>
          <div className="mt-5 grid gap-3 text-sm text-slate-300">
            {["Dose research", "RAG + Tavily evidence", "Calculator", "Structured answer"].map(
              (step, i) => (
                <div key={step} className="flex items-center gap-3 rounded-2xl bg-white/[0.04] p-3">
                  <span className="grid h-8 w-8 place-items-center rounded-full bg-cyan-300/10 text-cyan-100">
                    {i + 1}
                  </span>
                  {step}
                </div>
              )
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {features.map((feature) => (
          <article key={feature.title} className="panel-soft p-5">
            <h2 className="text-lg font-semibold text-white">{feature.title}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-400">{feature.text}</p>
          </article>
        ))}
      </section>

      <section className="panel p-6 sm:p-8">
        <p className="eyebrow">Academic prototype</p>
        <div className="mt-4 grid gap-6 lg:grid-cols-3">
          <p className="text-slate-300 lg:col-span-2">
            PeptiScout is designed as a transparent comparison system: GPT-only baselines,
            reasoning-style prompting, and a full agent with deterministic tools can be measured
            side by side across dosage, pathway, safety, and citation accuracy.
          </p>
          <div className="flex flex-wrap gap-2">
            {["FastAPI", "React", "LangGraph", "Pinecone", "Tavily", "OpenAI"].map((tag) => (
              <span key={tag} className="metric-pill">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
