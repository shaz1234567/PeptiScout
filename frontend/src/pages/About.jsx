const stack = [
  "React + Vite + Tailwind frontend",
  "FastAPI backend",
  "OpenAI GPT models",
  "LangGraph agent orchestration",
  "Pinecone RAG",
  "Tavily search",
  "VLM image analysis",
];

export default function About() {
  return (
    <div className="space-y-8">
      <section className="panel p-6 sm:p-8">
        <p className="eyebrow">About the project</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">PeptiScout AI</h1>
        <p className="mt-4 max-w-3xl text-lg leading-8 text-slate-300">
          PeptiScout AI is an academic research assistant prototype for peptide-related
          information retrieval, structured answer synthesis, dosage calculation, source vetting,
          and evaluation across multiple model modes.
        </p>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <article className="panel-soft p-6">
          <h2 className="text-2xl font-semibold text-white">Academic Context</h2>
          <p className="mt-4 leading-7 text-slate-300">
            Built for CPS 5801 at Kean University, the project compares baseline prompting,
            retrieval-augmented generation, deterministic tools, and agentic orchestration for
            peptide research workflows.
          </p>
        </article>
        <article className="panel-soft p-6">
          <h2 className="text-2xl font-semibold text-white">Research Use Disclaimer</h2>
          <p className="mt-4 leading-7 text-slate-300">
            This application is for educational and research review only. It does not diagnose,
            treat, prescribe, or replace qualified clinical guidance.
          </p>
        </article>
      </section>

      <section className="panel p-6">
        <h2 className="text-2xl font-semibold text-white">Tech Stack</h2>
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {stack.map((item) => (
            <div key={item} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4 text-slate-300">
              {item}
            </div>
          ))}
        </div>
      </section>

      <section className="panel-soft p-6">
        <h2 className="text-2xl font-semibold text-white">Team</h2>
        <p className="mt-4 leading-7 text-slate-300">
          Developed as a student research prototype for PeptiScout AI. Team details can be updated
          as final project credits are prepared.
        </p>
      </section>
    </div>
  );
}
