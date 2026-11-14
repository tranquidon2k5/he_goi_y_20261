"use client";

import { useState } from "react";

/** Chấm 0.5–5 sao, bước 0.5. Bàn phím: ←/→ đổi nửa sao, Enter để lưu. */
export function Stars({ value, onChange, small = false, readOnly = false }:
  { value: number; onChange?: (v: number) => void; small?: boolean; readOnly?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const [kb, setKb] = useState<number | null>(null);
  const shown = hover ?? kb ?? value;
  const at = (e: React.MouseEvent<HTMLSpanElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return Math.max(0.5, Math.min(5, Math.ceil(((e.clientX - r.left) / r.width) * 10) / 2));
  };
  if (readOnly || !onChange) {
    return (
      <span className={`stars ro ${small ? "sm" : ""}`} aria-label={`${value} sao`}>
        <span className="bgs">★★★★★</span><span className="fgs" style={{ width: `${(value / 5) * 100}%` }}>★★★★★</span>
      </span>
    );
  }
  return (
    <span
      className={`stars ${small ? "sm" : ""}`}
      role="slider" tabIndex={0} aria-label="Chấm sao" aria-valuemin={0.5} aria-valuemax={5} aria-valuenow={shown || undefined}
      aria-valuetext={shown ? `${shown} sao` : "chưa chấm"}
      onMouseMove={(e) => setHover(at(e))}
      onMouseLeave={() => setHover(null)}
      onClick={(e) => { const v = at(e); setHover(null); onChange(v); }}
      onKeyDown={(e) => {
        const step = (d: number) => setKb((k) => Math.max(0.5, Math.min(5, (k ?? (value || 0)) + d)));
        if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); step(0.5); }
        else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); step(-0.5); }
        else if (e.key === "Enter" && kb) { onChange(kb); setKb(null); }
      }}
      onBlur={() => setKb(null)}
    >
      <span className="bgs">★★★★★</span><span className="fgs" style={{ width: `${(shown / 5) * 100}%` }}>★★★★★</span>
    </span>
  );
}
