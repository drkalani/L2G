const STORAGE_KEY = "l2g-api-base";
const DEFAULT_API_BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/$/, "");
const FALLBACK_API_BASE = "/api";
const API_BASE_CHANGED = "l2g:api-base-changed";

const sanitizeBase = (value: string): string => {
  let base = (value || "").trim().replace(/\/$/, "");
  if (!base) {
    return DEFAULT_API_BASE || FALLBACK_API_BASE;
  }
  if (base.startsWith("/")) {
    return base || FALLBACK_API_BASE;
  }
  if (!/^[a-z][a-z0-9+\-.]*:\/\//i.test(base)) {
    base = `https://${base}`;
  }
  return base;
};

const getStoredApiBase = (): string => {
  if (typeof window === "undefined") {
    return sanitizeBase(DEFAULT_API_BASE);
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && stored.trim()) {
      return sanitizeBase(stored);
    }
  } catch {
    // storage may be unavailable in some environments
  }
  return sanitizeBase(DEFAULT_API_BASE);
};

let runtimeApiBase = getStoredApiBase();

const normalizePath = (path: string): string => {
  return path.startsWith("/") ? path : `/${path}`;
};

const buildApiUrl = (path: string): string => `${getApiBase()}${normalizePath(path)}`;

/** Free ngrok tunnels may return an HTML interstitial (ERR_NGROK_6024) unless this header is sent. */
const isNgrokRequestUrl = (url: string): boolean => {
  try {
    const u = new URL(url, typeof window !== "undefined" ? window.location.origin : "https://local.invalid");
    return /ngrok/i.test(u.hostname);
  } catch {
    return /ngrok/i.test(url);
  }
};

const apiFetch = (input: string, init?: RequestInit): Promise<Response> => {
  if (!isNgrokRequestUrl(input)) {
    return fetch(input, init);
  }
  const headers = new Headers(init?.headers);
  if (!headers.has("ngrok-skip-browser-warning")) {
    headers.set("ngrok-skip-browser-warning", "true");
  }
  return fetch(input, { ...init, headers });
};

export const getApiBase = (): string => runtimeApiBase;

export const setApiBase = (candidate: string): string => {
  const next = sanitizeBase(candidate);
  runtimeApiBase = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, next);
    window.dispatchEvent(new Event(API_BASE_CHANGED));
  } catch {
    // no-op if storage is not available
  }
  return next;
};

export const resetApiBase = (): string => setApiBase(DEFAULT_API_BASE);

export const apiBaseDidChange = (listener: () => void): (() => void) => {
  if (typeof window === "undefined") {
    return () => {};
  }
  const handler = () => listener();
  window.addEventListener(API_BASE_CHANGED, handler);
  return () => window.removeEventListener(API_BASE_CHANGED, handler);
};

export const pingBackend = (candidateBase?: string): Promise<{ status: string }> => {
  const targetBase = candidateBase ? sanitizeBase(candidateBase) : getApiBase();
  const url = `${targetBase}/health`;
  return parse<{ status: string }>(apiFetch(url));
};

export const isColabCandidate = (value: string): boolean => isNgrokRequestUrl(sanitizeBase(value));

async function parse<T>(res: Response | Promise<Response>): Promise<T> {
  const response = await res;
  if (!response.ok) {
    const t = await response.text();
    throw new Error(t || response.statusText);
  }
  return response.json() as Promise<T>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return parse<T>(apiFetch(buildApiUrl(path), init));
}

async function triggerDownload(path: string, fallbackName: string): Promise<void> {
  const response = await apiFetch(buildApiUrl(path));
  if (!response.ok) {
    const t = await response.text();
    throw new Error(t || response.statusText);
  }
  const blob = await response.blob();
  const cd = response.headers.get("Content-Disposition");
  let name = fallbackName;
  if (cd) {
    const m = /filename="?([^";]+)"?/i.exec(cd);
    if (m?.[1]) name = m[1];
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export type Processor = "auto" | "cuda" | "mps" | "cpu";
export type PipelineMode = "classify" | "ner" | "normalize" | "full";
export type NerMethod = "transformers" | "bent";

export interface Project {
  id: string;
  name: string;
  disease_key: string;
  description: string;
}

export interface DeviceInfo {
  available: { cuda: boolean; mps: boolean; cpu: boolean };
  recommended: string;
}

export interface JobRecord {
  job_id: string;
  state: "queued" | "running" | "completed" | "failed";
  message: string;
  project_id: string | null;
  created_at: string;
  updated_at: string;
  progress?: number | null;
  result?: Record<string, unknown> | null;
}

export interface BaseModelCatalog {
  models: string[];
}

export interface BaseModelDownloadResponse {
  model_id: string;
  downloaded: boolean;
  status: string;
  message: string;
}

export interface ModelCompatibilityResult {
  model_id: string;
  expected_task: "classification" | "token_classification";
  compatible: boolean;
  detected_tasks: string[];
  message: string;
}

export interface ProjectModelInfo {
  model_id: string;
  path: string;
}

export interface ProjectModelCatalog {
  models: ProjectModelInfo[];
}

export interface LastRunInfo {
  path: string | null;
  files: string[];
}

export type ExportArtifact =
  | "classification"
  | "mentions"
  | "normalized"
  | "bundle";
export type ExportFormat = "csv" | "xlsx" | "pkl";

export interface TrainingConfig {
  processor: Processor;
  base_model: string;
  learning_rate: number;
  num_train_epochs: number;
  per_device_train_batch_size: number;
  per_device_eval_batch_size: number;
  weight_decay: number;
  seed: number;
  max_length: number;
  fp16: boolean | null;
}

export interface Article {
  pmid: string;
  text: string;
  /** When set with abstract in `text`, backend uses BERT pair encoding (DKDM-style). */
  title?: string | null;
  label?: number | null;
}

export interface ImportStats {
  total_rows: number;
  imported_rows: number;
  skipped_rows: number;
  skipped_missing_pmid: number;
  skipped_missing_text_or_title: number;
  skipped_duplicates: number;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  devices: () => request<DeviceInfo>("/devices"),

  listProjects: () => request<Project[]>("/projects"),

  createProject: (body: {
    name: string;
    disease_key: string;
    description?: string;
  }) =>
    request<Project>("/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  listModels: (projectId: string) =>
    request<{ models: string[] }>(`/projects/${projectId}/models`),

  listProjectModelCatalog: (projectId: string) =>
    request<ProjectModelCatalog>(`/projects/${projectId}/models/catalog`),

  trainRelevance: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig,
    validation_split: number
  ) =>
    request<{ job_id: string; state: string }>(`/train/${projectId}/relevance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        articles,
        config,
        validation_split,
      }),
    }),

  trainKfold: (
    projectId: string,
    articles: Article[],
    config: TrainingConfig & { n_splits: number }
  ) =>
    request<{ job_id: string; state: string }>(`/train/${projectId}/relevance/kfold`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ articles, config }),
    }),

  jobStatus: (jobId: string) =>
    request<{
      job_id: string;
      state: string;
      message: string;
      created_at: string;
      updated_at: string;
      progress?: number | null;
      result: unknown;
    }>(`/train/jobs/${jobId}`),

  listJobs: (projectId?: string, limit?: number) => {
    const params = new URLSearchParams();
    if (projectId) {
      params.set("project_id", projectId);
    }
    if (typeof limit === "number" && Number.isInteger(limit) && limit > 0) {
      params.set("limit", `${limit}`);
    }
    const query = params.toString();
    return request<JobRecord[]>(`/train/jobs${query ? `?${query}` : ""}`);
  },

  lastRun: (projectId: string) =>
    request<LastRunInfo>(`/projects/${projectId}/data/last-run`),

  importArticles: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiFetch(buildApiUrl(`/projects/${projectId}/data/import/articles`), {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json() as Promise<{
      kind: string;
      row_count: number;
      articles: Article[];
      import_stats?: ImportStats;
    }>;
  },

  importMentions: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiFetch(buildApiUrl(`/projects/${projectId}/data/import/mentions`), {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json() as Promise<{
      kind: string;
      row_count: number;
      mentions: Record<string, unknown>[];
    }>;
  },

  downloadExport: (
    projectId: string,
    artifact: ExportArtifact,
    format: ExportFormat,
    fallbackName: string
  ) =>
    triggerDownload(`/projects/${projectId}/data/export/${artifact}?format=${format}`, fallbackName),

  downloadTemplate: (projectId: string, kind: "articles" | "mentions") =>
    triggerDownload(
      `/projects/${projectId}/data/templates/${kind}`,
      kind === "articles" ? "articles_template.csv" : "mentions_template.csv"
    ),

  listBaseModels: () => request<BaseModelCatalog>("/models/base"),

  downloadBaseModel: (modelId: string) =>
    request<BaseModelDownloadResponse>("/models/base/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    }),

  validateModelTask: (
    modelId: string,
    expectedTask: "classification" | "token_classification"
  ) => {
    const p = new URLSearchParams({ model_id: modelId, expected_task: expectedTask });
    return request<ModelCompatibilityResult>(`/models/validate?${p.toString()}`);
  },

  runPipeline: (body: {
    project_id: string;
    model_id: string;
    articles: Article[];
    mode: PipelineMode;
    processor: Processor;
    ner_model: string;
    ner_method: NerMethod;
    bent_service_url?: string;
    batch_size: number;
    use_wikipedia_fallback: boolean;
    mentions_json?: Record<string, unknown>[];
  }) =>
    request<Record<string, unknown>>("/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  pubmedFetch: (
    projectId: string,
    body: {
      email: string;
      query?: string;
      pmids?: string[];
      max_results?: number;
      retstart?: number;
      min_abstract_chars?: number;
      sleep_between_batches?: number;
    }
  ) =>
    request<{
      queried_id_count: number;
      search_total_estimate: number | null;
      row_count: number;
      articles: { pmid: string; title: string; text: string }[];
    }>(`/projects/${projectId}/data/pubmed/fetch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  importLitSuggestScores: async (projectId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await apiFetch(buildApiUrl(`/projects/${projectId}/data/import/litsuggest-scores`), {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json() as Promise<{
      kind: string;
      row_count: number;
      litsuggest: { pmid: string; score: number }[];
    }>;
  },

  compareLitSuggest: (
    projectId: string,
    body: {
      primary: Record<string, unknown>[];
      litsuggest: Record<string, unknown>[];
      score_threshold: number;
    }
  ) =>
    request<Record<string, unknown>>(`/projects/${projectId}/data/compare/litsuggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
