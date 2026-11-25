import { useState, useEffect } from 'react';
import { Activity, Radio, Cpu, ShieldAlert, Crosshair, Check, Minus } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Spread state from the live z-score. Describes where the spread sits relative
// to its own history — not an instruction to trade (AGENTS.md rule 11).
const SPREAD_STATE = {
  LONG_A_SHORT_B: { label: 'Below −2σ', tone: 'text-[#a2d0c0] border-[#a2d0c0]/30 bg-[#255144]/40' },
  SHORT_A_LONG_B: { label: 'Above +2σ', tone: 'text-[#ffb4ab] border-[#ffb4ab]/30 bg-[#93000a]/40' },
  STOP: { label: 'Beyond ±4σ', tone: 'text-[#ffb4ab] border-[#ffb4ab]/30 bg-[#93000a]/20' },
  EXIT: { label: 'Near mean', tone: 'text-[#9a8f81] border-[#4e453a]/30 bg-[#3a342a]/40' },
  NEUTRAL: { label: 'Neutral', tone: 'text-[#9a8f81] border-[#4e453a]/30 bg-[#3a342a]/40' },
};

const fmt = (v, digits = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—');

async function getJson(url) {
  const res = await apiFetch(url);
  const body = await res.json();
  if (!res.ok) throw new Error(body?.detail?.message || `${url}: HTTP ${res.status}`);
  return body;
}

export default function QuantSignals() {
  const [pairs, setPairs] = useState(null);
  const [regime, setRegime] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const load = async () => {
      try {
        const [pairsData, regimeData] = await Promise.all([
          getJson('/api/quant/pairs'),
          getJson('/api/quant/regime'),
        ]);
        if (cancelled) return;
        setPairs(pairsData);
        setRegime(regimeData);
        setError(null);
        if (pairsData?.refreshing) timer = setTimeout(load, 15000);
      } catch (e) {
        console.error(e);
        if (!cancelled) setError(e.message || 'Quant engine unavailable.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] bg-[#0E1A16] text-[#94c2b2]">
        <Radio size={32} className="animate-pulse mb-4 text-[#255144]" />
        <p className="font-mono text-xs uppercase tracking-widest animate-pulse">Computing on live NSE data…</p>
        <p className="font-mono text-[10px] mt-2 opacity-70">First run downloads ~5 years for the universe (≈30s)</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[60vh] bg-[#0E1A16] text-[#ffb4ab]">
        <ShieldAlert size={32} className="mb-4 text-[#ba1a1a]" />
        <p className="font-mono text-xs uppercase tracking-widest">{error}</p>
      </div>
    );
  }

  const marketState = regime?.current_regime || 'UNKNOWN';
  const isBull = marketState === 'BULL';
  const isBear = marketState === 'BEAR';
  const rows = pairs?.pairs ?? [];
  const status = pairs?.stale ? (pairs?.refreshing ? 'REFRESHING' : 'STALE') : pairs?.provisional ? 'PROVISIONAL' : 'CURRENT';

  return (
    <div className="h-full flex flex-col bg-[#0E1A16] text-[#ebe1d3] overflow-y-auto md:overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-[#255144]/30 bg-[#13231E] flex flex-wrap gap-4 justify-between items-center">
        <div className="flex items-center gap-4">
          <Cpu className="text-[#a2d0c0]" size={20} />
          <div>
            <h2 className="font-mono text-lg tracking-widest text-[#a2d0c0] uppercase">Quant Engine Terminal</h2>
            <p className="font-mono text-[10px] text-[#94c2b2] uppercase opacity-70">Subsystem: Stat-Arb & Regime Detection</p>
          </div>
        </div>
        <div className="flex gap-6">
          <div className="flex flex-col items-end">
            <span className="font-mono text-[10px] text-[#94c2b2] uppercase">Data as of</span>
            <span className="font-mono text-xs text-[#a2d0c0]">{pairs?.as_of ?? '—'}</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="font-mono text-[10px] text-[#94c2b2] uppercase">Status</span>
            <span className="font-mono text-xs text-[#a2d0c0] flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${status === 'CURRENT' ? 'bg-[#a2d0c0]' : 'bg-[#9a8f81] animate-pulse'}`}></span>
              {status}
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 md:p-6 grid grid-cols-12 gap-6 md:overflow-hidden">
        {/* Left column */}
        <div className="col-span-12 md:col-span-4 flex flex-col gap-6 md:overflow-y-auto md:pr-2">
          <div className="border border-[#255144]/50 bg-[#13231E] p-5 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2 opacity-10"><Activity size={120} /></div>
            <h3 className="font-mono text-[10px] uppercase tracking-widest text-[#94c2b2] mb-4 flex items-center gap-2">
              <Crosshair size={12} /> HMM Regime Detector (NIFTY 50)
            </h3>
            <div className="mb-6">
              <p className="font-mono text-[10px] uppercase text-[#94c2b2] mb-1">Model State · {regime?.as_of ?? '—'}</p>
              <div className={`font-display text-4xl tracking-tight ${isBull ? 'text-[#a2d0c0]' : isBear ? 'text-[#ffb4ab]' : 'text-[#ebe1d3]'}`}>
                {marketState}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 border-t border-[#255144]/30 pt-4">
              <div>
                <p className="font-mono text-[10px] uppercase text-[#94c2b2]">P(leave next session)</p>
                <p className="font-mono text-sm">
                  {typeof regime?.prob_leave_next_session === 'number' ? `${(regime.prob_leave_next_session * 100).toFixed(1)}%` : '—'}
                </p>
              </div>
              <div>
                <p className="font-mono text-[10px] uppercase text-[#94c2b2]">Time in state</p>
                <p className="font-mono text-sm">{regime?.sessions_in_state != null ? `${regime.sessions_in_state} sessions` : '—'}</p>
                {regime?.state_since && <p className="font-mono text-[10px] text-[#94c2b2]">since {regime.state_since}</p>}
              </div>
            </div>
          </div>

          <div className="border border-[#255144]/50 bg-[#13231E] flex flex-col">
            <div className="border-b border-[#255144]/30 p-3 bg-[#0E1A16]">
              <span className="font-mono text-[10px] uppercase tracking-widest text-[#94c2b2]">Stat-Arb Diagnostics</span>
            </div>
            <table className="w-full text-left font-mono text-xs">
              <tbody>
                {[
                  ['Pairs tested (train)', pairs?.total_scanned],
                  ['Cointegrated (p < 0.05)', pairs?.cointegrated],
                  ['OOS backtested', pairs?.backtested],
                  ['Passed OOS gate', pairs?.validated],
                ].map(([label, val], idx) => (
                  <tr key={label} className={`border-b border-[#255144]/30 ${idx % 2 === 0 ? 'bg-[#13231E]' : 'bg-[#0E1A16]'}`}>
                    <td className="p-3 text-[#94c2b2] uppercase text-[10px]">{label}</td>
                    <td className="p-3 text-right text-[#a2d0c0]">{val ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {pairs?.total_scanned > 0 && (
              <p className="p-3 font-mono text-[10px] text-[#94c2b2] leading-relaxed">
                At p &lt; 0.05, about {Math.round(pairs.total_scanned * 0.05)} of {pairs.total_scanned} pairs would look
                cointegrated by chance alone. Only pairs that then pass the out-of-sample gate
                (Sharpe ≥ {pairs?.gate?.min_oos_sharpe}, max DD ≤ {(pairs?.gate?.max_oos_drawdown ?? 0) * 100}%,
                {' '}{pairs?.gate?.min_oos_sessions} sessions) are marked validated.
              </p>
            )}
            {pairs?.windows && (
              <p className="px-3 pb-3 font-mono text-[10px] text-[#94c2b2] opacity-70">
                Train {pairs.windows.train_start} → {pairs.windows.train_end} · OOS {pairs.windows.test_start} → {pairs.windows.test_end}
              </p>
            )}
          </div>
        </div>

        {/* Pair table */}
        <div className="col-span-12 md:col-span-8 border border-[#255144]/50 bg-[#13231E] flex flex-col overflow-hidden min-h-[320px]">
          <div className="border-b border-[#255144]/30 p-4 bg-[#0E1A16] flex justify-between items-center">
            <span className="font-mono text-[10px] uppercase tracking-widest text-[#a2d0c0]">
              Pair Spread States · {pairs?.as_of ?? '—'}
            </span>
            <span className="font-mono text-[10px] bg-[#255144]/30 text-[#94c2b2] px-2 py-1 rounded-sm border border-[#255144]/50">
              {pairs?.validated ?? 0} validated of {rows.length} backtested
            </span>
          </div>
          <div className="flex-1 overflow-auto">
          <div className="min-w-[640px]">
          <div className="grid grid-cols-7 gap-2 p-3 border-b border-[#255144]/30 bg-[#13231E] sticky top-0 z-10">
            {['Asset X', 'Asset Y', 'Z-Score', 'Half-Life', 'OOS Sharpe', 'OOS Gate', 'Spread'].map((h, i) => (
              <div key={h} className={`font-mono text-[10px] uppercase text-[#94c2b2] ${i >= 2 ? 'text-right' : ''}`}>{h}</div>
            ))}
          </div>
          <div>
            {rows.length > 0 ? rows.map((pair) => {
              const state = SPREAD_STATE[pair.signal_state] ?? SPREAD_STATE.NEUTRAL;
              return (
                <div key={pair.pair} className="grid grid-cols-7 gap-2 p-4 border-b border-[#255144]/10 hover:bg-[#255144]/20 transition-colors items-center">
                  <div className="font-mono text-sm text-[#ebe1d3]">{pair.x}</div>
                  <div className="font-mono text-sm text-[#ebe1d3]">{pair.y}</div>
                  <div className="font-mono text-sm text-right">
                    <span className={pair.z_score > 2 ? 'text-[#ffb4ab]' : pair.z_score < -2 ? 'text-[#a2d0c0]' : 'text-[#ebe1d3]'}>
                      {fmt(pair.z_score)}
                    </span>
                  </div>
                  <div className="font-mono text-xs text-right text-[#94c2b2]">{fmt(pair.half_life, 1)}d</div>
                  <div className="font-mono text-xs text-right text-[#94c2b2]">{fmt(pair.oos?.sharpe_ratio)}</div>
                  <div className="flex justify-end">
                    {pair.validated ? (
                      <span className="flex items-center gap-1 text-[#a2d0c0] font-mono text-[10px]"><Check size={10} /> PASS</span>
                    ) : (
                      <span className="flex items-center gap-1 text-[#9a8f81] font-mono text-[10px]"><Minus size={10} /> FAIL</span>
                    )}
                  </div>
                  <div className="flex justify-end">
                    <span className={`px-2 py-1 rounded-sm border font-mono text-[10px] ${state.tone}`}>{state.label}</span>
                  </div>
                </div>
              );
            }) : (
              <div className="p-8 text-center text-[#94c2b2] font-mono text-xs uppercase">No cointegrated pairs in the training window.</div>
            )}
          </div>
          </div>
          </div>
          {pairs?.disclaimer && (
            <p className="p-3 border-t border-[#255144]/30 font-mono text-[10px] text-[#94c2b2] opacity-70">{pairs.disclaimer}</p>
          )}
        </div>
      </div>
    </div>
  );
}
