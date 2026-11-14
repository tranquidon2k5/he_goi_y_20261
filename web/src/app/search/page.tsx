import { Suspense } from "react";
import { SearchView } from "./search-view";

export default function SearchPage() {
  return (
    <Suspense>
      <SearchView />
    </Suspense>
  );
}
