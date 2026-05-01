import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchResults } from "../lib/api.js";

const metricKeys = [
  ["mean_ds", "DS"],
  ["mean_ds_strict", "DS strict"],
  ["mean_pc", "PC"],
  ["mean_tsr", "TSR"],
  ["mean_ca", "CA"],
];

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function delta(value) {
  if (value === null || value === undefined) return "not available";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pts`;
}

export default function Results() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetchResults()
      .then((result) => {
        if (active) setData(result);
      })
      .catch((e) => {
        if (active) setError(e?.response?.data?.detail || e.message || "Unable to load results");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const rows = useMemo(() => {
    const table = data?.comparison_table || {};
    return Object.entries(table).map(([mode, values]) => ({ mode, ...values }));
  }, [data]);

  const chartRows = rows.map((row) => ({
    mode: row.mode.replace("baseline-", "base-").replace("full-agent", "agent"),
    DS: row.mean_ds,
    PC: row.mean_pc,
    TSR: row.mean_tsr,
    CA: row.mean_ca,
  }));

  const ab = data?.ablations || {};

  return (
    <div className="space-y-8">
      <section className="panel p-6 sm:p-8">
        <p className="eyebrow">Evaluation Results</p>
        <h1 className="mt-3 text-4xl font-semibold text-white">Mode comparison and ablations</h1>
        <p className="mt-4 max-w-3xl text-slate-300">
          Results are loaded from the FastAPI backend and reflect the current
          <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5">backend/data/results.json</code>
          evaluation export.
        </p>
        {data?.meta && (
          <div className="mt-5 flex flex-wrap gap-2">
            <span className="metric-pill">Modes: {data.meta.modes?.length || rows.length}</span>
            <span className="metric-pill">CA {data.meta.skip_ca ? "skipped" : "validated"}</span>
            <span className="metric-pill">{data.meta.timestamp?.slice(0, 10)}</span>
          </div>
        )}
      </section>

      {loading && <div className="panel p-8 text-slate-300">Loading results...</div>}
      {error && <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 p-4 text-rose-100">{error}</div>}

      {!loading && !error && (
        <>
          <section className="panel p-5">
            <h2 className="text-2xl font-semibold text-white">Comparison Table</h2>
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[780px] text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="py-3 pr-4">Mode</th>
                    {metricKeys.map(([, label]) => (
                      <th key={label} className="px-4 py-3">
                        {label}
                      </th>
                    ))}
                    <th className="px-4 py-3">n</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {rows.map((row) => (
                    <tr key={row.mode} className="text-slate-300">
                      <td className="py-4 pr-4 font-medium text-white">{row.mode}</td>
                      {metricKeys.map(([key]) => (
                        <td key={key} className="px-4 py-4">
                          {fmt(row[key])}
                        </td>
                      ))}
                      <td className="px-4 py-4 text-slate-500">{row.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel p-5">
            <h2 className="text-2xl font-semibold text-white">Metric Snapshot</h2>
            <div className="mt-5 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                  <XAxis dataKey="mode" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#94a3b8" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                  <Tooltip
                    contentStyle={{ background: "#020617", border: "1px solid rgba(255,255,255,0.12)" }}
                    formatter={(v) => fmt(v)}
                  />
                  <Legend />
                  <Bar dataKey="DS" fill="#22d3ee" />
                  <Bar dataKey="PC" fill="#818cf8" />
                  <Bar dataKey="TSR" fill="#34d399" />
                  <Bar dataKey="CA" fill="#fbbf24" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-3">
            <article className="panel-soft p-5">
              <h3 className="text-lg font-semibold text-white">Ablation A: Calculator</h3>
              <p className="mt-3 text-sm text-slate-400">
                Full agent vs calculator disabled: <span className="text-cyan-100">{delta(ab.A?.delta_ds)}</span>.
                Positive deltas suggest deterministic math improves dosage accuracy.
              </p>
            </article>
            <article className="panel-soft p-5">
              <h3 className="text-lg font-semibold text-white">Ablation B: Reasoning</h3>
              <p className="mt-3 text-sm text-slate-400">
                Full agent vs no-reasoning PC delta: <span className="text-cyan-100">{delta(ab.B?.delta_pc)}</span>.
                This measures pathway/cofactor alignment on MOA rows.
              </p>
            </article>
            <article className="panel-soft p-5">
              <h3 className="text-lg font-semibold text-white">Ablation C: RAG vs Fine-tune</h3>
              <p className="mt-3 text-sm text-slate-400">
                PC delta: <span className="text-cyan-100">{delta(ab.C?.delta_pc)}</span>. CA delta:{" "}
                <span className="text-cyan-100">{delta(ab.C?.delta_ca)}</span>. Null values mean the
                fine-tuned/no-rag run was not available.
              </p>
            </article>
          </section>
        </>
      )}
    </div>
  );
}
