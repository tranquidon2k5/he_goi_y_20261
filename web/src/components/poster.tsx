"use client";

import Image from "next/image";
import { useState } from "react";
import { GLYPH } from "@/lib/genres";
import type { Movie } from "@/lib/types";

function hue(s: string) {
  let h = 0;
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

/** Poster TMDB nếu có, không thì ảnh thay thế dựng từ tiêu đề (PRD mục 10: thiếu metadata TMDB). */
export function Poster({ movie, className = "", sizes = "170px", priority = false }:
  { movie: Movie; className?: string; sizes?: string; priority?: boolean }) {
  const [broken, setBroken] = useState(false);
  const h = hue(movie.title);
  const img = movie.poster_url && !broken;
  return (
    <div
      className={`poster ${img ? "has-img" : ""} ${className}`}
      style={{ "--pg": `linear-gradient(155deg,hsl(${h} 55% 38%),hsl(${(h + 40) % 360} 60% 16%))` } as React.CSSProperties}
    >
      {img && (
        <Image className="pimg" src={movie.poster_url!} alt={movie.title} fill sizes={sizes} priority={priority}
          onError={() => setBroken(true)} />
      )}
      <div className="glyph" aria-hidden>{GLYPH[movie.genres[0]] ?? "✦"}</div>
      <div className="pt">{movie.title}</div>
      <div className="py">{movie.year ?? "—"}{movie.genres[0] ? ` · ${movie.genres[0]}` : ""}</div>
    </div>
  );
}
