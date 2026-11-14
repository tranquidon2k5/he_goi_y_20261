import { Suspense } from "react";
import { MovieView } from "./movie-view";

export default function MoviePage() {
  return (
    <Suspense>
      <MovieView />
    </Suspense>
  );
}
