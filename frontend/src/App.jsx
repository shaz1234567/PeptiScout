import { NavLink, Route, Routes } from "react-router-dom";
import Architecture from "./pages/Architecture.jsx";
import Demo from "./pages/Demo.jsx";
import Home from "./pages/Home.jsx";
import Results from "./pages/Results.jsx";
import About from "./pages/About.jsx";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/demo", label: "Demo" },
  { to: "/architecture", label: "Architecture" },
  { to: "/results", label: "Results" },
  { to: "/about", label: "About" },
];

function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-cyan-400/10 bg-slate-950/80 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <NavLink to="/" className="group flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-2xl border border-cyan-300/30 bg-cyan-300/10 text-cyan-200 shadow-lg shadow-cyan-500/10">
            PS
          </div>
          <div>
            <p className="text-sm uppercase tracking-[0.35em] text-cyan-200/70">PeptiScout</p>
            <h1 className="text-lg font-semibold text-white">AI Research Console</h1>
          </div>
        </NavLink>
        <div className="flex flex-wrap gap-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                [
                  "rounded-full px-4 py-2 text-sm font-medium transition",
                  isActive
                    ? "bg-cyan-300 text-slate-950 shadow-lg shadow-cyan-400/20"
                    : "text-slate-300 hover:bg-white/10 hover:text-white",
                ].join(" ")
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>
    </header>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(34,211,238,0.18),transparent_35%),radial-gradient(circle_at_80%_20%,rgba(99,102,241,0.16),transparent_30%),linear-gradient(180deg,#020617,#0f172a_45%,#020617)]" />
      <Header />
      <main className="mx-auto max-w-7xl px-5 py-10 sm:py-14">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/results" element={<Results />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <footer className="border-t border-white/10 px-5 py-8 text-center text-sm text-slate-500">
        PeptiScout AI is an academic research prototype. Not medical advice.
      </footer>
    </div>
  );
}
