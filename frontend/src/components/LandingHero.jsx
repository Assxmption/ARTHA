import { ArrowRight } from 'lucide-react';

export default function LandingHero({ onEnter }) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center p-8 bg-surface text-on-surface relative overflow-hidden transition-colors duration-300">
      
      {/* Decorative Grid Background */}
      <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'linear-gradient(var(--outline) 1px, transparent 1px), linear-gradient(90deg, var(--outline) 1px, transparent 1px)', backgroundSize: '40px 40px' }}></div>

      <div className="relative z-10 w-full max-w-4xl text-center">
        
        <div className="mb-8">
          <span className="font-ui text-xs uppercase tracking-[0.2em] text-primary border border-primary px-4 py-1">
            Systematic Editorial
          </span>
        </div>

        <h1 className="font-display text-6xl md:text-8xl tracking-tight text-on-surface mb-8 leading-tight">
          The Architecture <br />
          <span className="italic text-primary">of Value.</span>
        </h1>

        <p className="font-body text-xl md:text-2xl text-on-surface-variant max-w-2xl mx-auto mb-16 leading-relaxed">
          Institutional-grade quantitative regimes paired with discretionary editorial narratives for Indian equities and commodities.
        </p>

        <button 
          onClick={onEnter}
          className="group flex items-center gap-4 mx-auto border border-on-surface px-8 py-4 font-ui uppercase tracking-widest text-sm hover:bg-on-surface hover:text-inverse-on-surface transition-all duration-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        >
          Enter the Atelier
          <ArrowRight size={16} className="transform group-hover:translate-x-2 transition-transform duration-300" />
        </button>

      </div>

      <div className="absolute bottom-8 left-8">
        <p className="font-mono text-xs text-outline">LAT: 19.0760° N<br/>LON: 72.8777° E</p>
      </div>

      <div className="absolute bottom-8 right-8 text-right">
        <p className="font-mono text-xs text-outline">V. 2.0.0<br/>QUANT CORE ACTIVE</p>
      </div>
      
    </div>
  );
}
