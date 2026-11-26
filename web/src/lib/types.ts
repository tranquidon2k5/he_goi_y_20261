export type Movie = {
  id: number;
  title: string;
  year: number | null;
  genres: string[];
  overview: string | null;
  poster_url: string | null;
  backdrop_url: string | null;
  trailer_key: string | null;
  runtime: number | null;
  directors: string | null;
  cast: string | null;
  rating_avg: number;
  rating_count: number;
  is_new: boolean;
  imdb_id: string | null;
  tmdb_id: number | null;
};

export type User = {
  id: number;
  email: string;
  display_name: string;
  is_admin: boolean;
  onboarded: boolean;
  genres: string[];
  n_ratings: number;
};

export type Reason = { movie_id: number; title: string; rating?: number | null; shared_genres?: string[] };
export type RecItem = { movie: Movie; reason?: Reason };
export type RecRow = {
  key: string;
  kind: "for_you" | "because" | "trending" | "genre" | "new";
  title: string;
  subtitle: string;
  items: RecItem[];
};
export type HomeRecs = {
  request_id: string;
  model_version: string;
  cold: boolean;
  n_interactions: number;
  threshold: number;
  hero: RecItem | null;
  rows: RecRow[];
};

export type MovieDetail = {
  movie: Movie;
  me: {
    rating: number | null;
    rated_at: number | null;
    tags: { id: number; tag: string }[];
    watchlist: boolean;
    not_interested: boolean;
    watched: boolean;
  } | null;
  popular_tags: string[];
};

export type SearchResult = {
  total: number;
  page: number;
  page_size: number;
  items: { movie: Movie; my_rating: number | null }[];
};

export type ModelVersion = {
  version: string;
  recommender: string;
  kind: "refresh" | "finetune" | "full";
  status: "queued" | "running" | "live" | "pass" | "fail" | "archived";
  metrics: Record<string, number>;
  config: Record<string, unknown>;
  data_info: Record<string, unknown>;
  notes: string | null;
  created_at: number;
  deployed_at: number | null;
};

export type AdminMovies = { total: number; page: number; page_size: number; items: Movie[]; tmdb_enabled: boolean };

export type LabInfo = {
  live: ModelVersion | null;
  versions: ModelVersion[];
  recommenders: string[];
  stats: { p95_ms: number | null; pending_interactions: number; app_users: number; app_ratings: number; events: number };
  online: { row: string; impressions: number; clicks: number; ctr: number }[];
  offline: ({ version: string; recommender: string } & Record<string, number | string>)[];
  experiments: Experiments | null;
};

type MeanStd = [number, number];
export type ExperimentRow = Record<"ndcg@10" | "recall@10" | "ndcg@20" | "recall@20" | "mrr" | "coverage@10" | "diversity@10" | "novelty@10", MeanStd> & {
  seeds: number;
  n_trials: number;
  best_epoch: number;
  "val_ndcg@10": number;
  hp: Record<string, unknown>;
  by_history_len: Record<string, number>;
  by_target_popularity: Record<string, number>;
  "mmr_0.2": Record<string, number>;
  train_seconds: number;
};
export type Experiments = {
  models: Record<string, ExperimentRow>;
  labels: Record<string, string>;
  reference: Record<string, Record<string, number>>;
  significance?: { vs: string; t: number; p: number; mean_diff: number };
  ablation_tests?: Record<string, { t: number; p: number }>;
  cl_controlled?: { t: number; p: number; mean_diff: number };
  significance_v2?: { vs: string; t: number; p: number; mean_diff: number };
  rerank?: Record<string, { days: number; w: number; val_base: number; val_rerank: number; seeds: number[];
    test_base: number; test_rerank: number; test_rerank_std: number; by_history_len: Record<string, number> }>;
  best_on_val?: string;
  data: Record<string, unknown> & { users: number; items: number; interactions: number; since_year: number };
};
