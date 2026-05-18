import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  type Article,
  type LastRunInfo,
  type ImportStats,
  type DeviceInfo,
  type ProjectModelCatalog,
  type JobRecord,
  type ModelCompatibilityResult,
  type PipelineMode,
  type Processor,
  type TrainingConfig,
  type NerMethod,
  apiBaseDidChange,
  getApiBase,
  pingBackend,
  setApiBase,
} from "../api";

const defaultTrainConfig = (): TrainingConfig => ({
  processor: "auto",
  base_model: "dmis-lab/biobert-v1.1",
  learning_rate: 2e-5,
  num_train_epochs: 4,
  per_device_train_batch_size: 16,
  per_device_eval_batch_size: 16,
  weight_decay: 0.01,
  seed: 42,
  max_length: 512,
  fp16: null,
});

const mpsTrainPreset = (): TrainingConfig => ({
  processor: "mps",
  base_model: "dmis-lab/biobert-v1.1",
  learning_rate: 2e-5,
  num_train_epochs: 3,
  per_device_train_batch_size: 4,
  per_device_eval_batch_size: 4,
  weight_decay: 0.01,
  seed: 42,
  max_length: 256,
  fp16: false,
});

const exampleArticlesJson = `[
  {"pmid":"10000001","title":"Gene regulation in DKD","text":"TGF-beta signaling in diabetic nephropathy and kidney fibrosis.","label":1},
  {"pmid":"10000002","text":"Weather patterns in coastal regions unrelated to nephrology.","label":0}
]`;
const examplePipelineArticlesJson = `[
  {"pmid":"10000001","title":"Gene regulation in DKD","text":"TGF-beta signaling in diabetic nephropathy and kidney fibrosis."},
  {"pmid":"10000002","text":"Weather patterns in coastal regions unrelated to nephrology."}
]`;

const exampleComparePrimary = `[
  {"pmid":"10000001","label":1},
  {"pmid":"10000002","label":0}
]`;
const exampleCompareLitSuggest = `[
  {"pmid":"10000001","score":0.82},
  {"pmid":"10000002","score":0.31}
]`;

type ModelCheckState = {
  modelId: string;
  checking: boolean;
  compatible: boolean | null;
  detectedTasks: string[];
  message: string;
};

type ClassificationRow = {
  pmid?: string;
  text?: string;
  relevant?: number;
  relevance_prob?: number;
};

type WorkspaceStep = "backend" | "train" | "integrations" | "pipeline";

const makeProjectJobStateKey = (projectId: string) => `l2g-job-state:${projectId}`;
const makeProjectWorkspaceStateKey = (projectId: string) =>
  `l2g-workspace-state:${projectId}`;
const WORKSPACE_STATE_VERSION = 1;

const isWorkspaceStep = (value: unknown): value is WorkspaceStep => {
  return (
    value === "backend" ||
    value === "train" ||
    value === "integrations" ||
    value === "pipeline"
  );
};

const isPipelineMode = (value: unknown): value is PipelineMode => {
  return (
    value === "full" ||
    value === "classify" ||
    value === "ner" ||
    value === "normalize"
  );
};
const isNerMethod = (value: unknown): value is NerMethod => {
  return value === "transformers" || value === "bent";
};

const isProcessor = (value: unknown): value is Processor => {
  return value === "auto" || value === "cuda" || value === "mps" || value === "cpu";
};

const toStoredString = (value: unknown, fallback: string): string =>
  typeof value === "string" ? value : fallback;

const toStoredNumber = (value: unknown, fallback: number): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const toStoredBoolean = (value: unknown, fallback: boolean): boolean =>
  typeof value === "boolean" ? value : fallback;

const toStoredNumberFromString = (value: unknown, fallback: number): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  return fallback;
};

type PersistedWorkspaceState = {
  version: number;
  articlesJson: string;
  pipeJson: string;
  normJson: string;
  trainCfg: TrainingConfig;
  valSplit: number;
  kfoldSplits: number;
  modelId: string;
  mode: PipelineMode;
  processor: Processor;
  nerModel: string;
  nerMethod: NerMethod;
  bentServiceUrl: string;
  pipeCompareMethods: boolean;
  pipeBatchSize: number;
  pipeUseWikipedia: boolean;
  relevanceThreshold: number;
  baseModelToDownload: string;
  comparePrimaryJson: string;
  compareLitJson: string;
  compareThreshold: number;
  pubmedEmail: string;
  pubmedQuery: string;
  pubmedMax: number;
  pubmedMinAbstract: number;
  activeStep: WorkspaceStep;
};

type StoredTrainingConfig = Partial<Record<keyof TrainingConfig, unknown>>;

const toNumberOrNull = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const toPercentOrNull = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    const normalized = value <= 1 ? value * 100 : value;
    return Math.max(0, Math.min(100, normalized));
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      const normalized = parsed <= 1 ? parsed * 100 : parsed;
      return Math.max(0, Math.min(100, normalized));
    }
  }
  return null;
};

const isJobStateRunning = (state: unknown): boolean => {
  if (typeof state !== "string") return false;
  const normalized = state.toLowerCase();
  return (
    normalized === "running" ||
    normalized === "queued" ||
    normalized === "pending" ||
    normalized === "starting" ||
    normalized === "started" ||
    normalized === "in_progress"
  );
};

const isJobStateTerminal = (state: unknown): boolean => {
  if (typeof state !== "string") return false;
  const normalized = state.toLowerCase();
  return normalized === "completed" || normalized === "failed";
};

const toEpochMsOrNull = (value: unknown): number | null => {
  if (typeof value !== "string" || !value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const formatSecondsAsHuman = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return "0s";
  const rounded = Math.round(seconds);
  const h = Math.floor(rounded / 3600);
  const m = Math.floor((rounded % 3600) / 60);
  const s = rounded % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

const jobStateText = (state: string) => {
  if (!state) return "running";
  return state;
};

const getClassificationRows = (
  result: Record<string, unknown>
): ClassificationRow[] => {
  if (result.kind === "full") {
    const fullRows = result.classification;
    if (Array.isArray(fullRows)) {
      return fullRows as ClassificationRow[];
    }
    return [];
  }
  if (result.kind === "classification") {
    const rows = result.rows;
    if (Array.isArray(rows)) {
      return rows as ClassificationRow[];
    }
  }
  return [];
};

const getMentionsRows = (result: Record<string, unknown> | null): Record<string, unknown>[] => {
  if (!result) return [];
  if (result.kind === "full") {
    const mentions = result.mentions;
    if (Array.isArray(mentions)) {
      return mentions as Record<string, unknown>[];
    }
    return [];
  }
  if (result.kind === "ner" || result.kind === "mentions") {
    const rows = result.rows;
    if (Array.isArray(rows)) return rows as Record<string, unknown>[];
  }
  if (Array.isArray(result.rows)) {
    return result.rows as Record<string, unknown>[];
  }
  return [];
};

const getErrorDetailMessage = (message: string): string => {
  const source = message || "";
  if (!source.trim()) return "";
  const trimmed = source.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (typeof parsed === "object" && parsed !== null && "detail" in parsed) {
        const detail = (parsed as { detail?: unknown }).detail;
        if (typeof detail === "string") {
          return detail;
        }
      }
    } catch {
      return trimmed;
    }
  }
  return trimmed;
};

const isBentUnavailableError = (message: string): boolean => {
  const detail = getErrorDetailMessage(message);
  return (
    detail.includes("Bent method is selected but `bent` is not installed") ||
    detail.includes("Bent service request failed") ||
    detail.includes("Bent execution failed: Bent method requires `bent` package") ||
    detail.includes("requires `bent` package") ||
    detail.includes("Bent service returned status") ||
    detail.includes("BENT service at ") ||
    detail.includes("unavailable or returned an error")
  );
};

const formatImportNotice = (label: string, stats?: ImportStats | null) => {
  if (!stats) {
    return `${label}: import complete.`;
  }
  if (stats.imported_rows === 0) {
    return `${label}: no valid rows imported.`;
  }
  const bits = [`${stats.imported_rows} imported from ${stats.total_rows} rows`];
  const skipped: string[] = [];
  if (stats.skipped_missing_pmid > 0) {
    skipped.push(`${stats.skipped_missing_pmid} missing PMID`);
  }
  if (stats.skipped_missing_text_or_title > 0) {
    skipped.push(`${stats.skipped_missing_text_or_title} missing title/text`);
  }
  if (stats.skipped_duplicates > 0) {
    skipped.push(`${stats.skipped_duplicates} duplicates`);
  }
  if (skipped.length) {
    bits.push(`skipped: ${skipped.join(", ")}`);
  }
  return `${label}: ${bits.join(" · ")}`;
};

export function ProjectWorkspace() {
  const { id } = useParams<{ id: string }>();
  const projectId = id ?? "";

  const [models, setModels] = useState<string[]>([]);
  const [projectModels, setProjectModels] = useState<ProjectModelCatalog["models"]>([]);
  const [availableBaseModels, setAvailableBaseModels] = useState<string[]>([]);
  const [devices, setDevices] = useState<string | null>(null);
  const [trainCfg, setTrainCfg] = useState<TrainingConfig>(defaultTrainConfig);
  const [valSplit, setValSplit] = useState(0.2);
  const [kfoldSplits, setKfoldSplits] = useState(5);
  const [articlesJson, setArticlesJson] = useState(exampleArticlesJson);
  const [backendUrl, setBackendUrl] = useState(getApiBase());
  const [bentServiceUrl, setBentServiceUrl] = useState("");
  const [backendStatus, setBackendStatus] = useState<
    "pending" | "checking" | "connected" | "failed"
  >("pending");
  const [backendStatusText, setBackendStatusText] = useState("Not checked");
  const [backendMessage, setBackendMessage] = useState(
    "Test backend before running training or pipeline jobs."
  );
  const [backendDevices, setBackendDevices] = useState<DeviceInfo | null>(null);
  const [backendDevicesStatus, setBackendDevicesStatus] = useState<
    "pending" | "checking" | "done" | "error"
  >("pending");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobPoll, setJobPoll] = useState<
    JobRecord | Record<string, unknown> | null
  >(null);
  const [jobHistory, setJobHistory] = useState<JobRecord[]>([]);
  const [showOnlyJobsWithResult, setShowOnlyJobsWithResult] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [importNotice, setImportNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [isDownloadingBaseModel, setIsDownloadingBaseModel] = useState(false);
  const [baseModelToDownload, setBaseModelToDownload] = useState("");
  const [activeStep, setActiveStep] = useState<WorkspaceStep>("backend");

  const [modelId, setModelId] = useState("");
  const [mode, setMode] = useState<PipelineMode>("full");
  const [pipeJson, setPipeJson] = useState(examplePipelineArticlesJson);
  const [processor, setProcessor] = useState<Processor>("auto");
  const [nerModel, setNerModel] = useState("pruas/BENT-PubMedBERT-NER-Gene");
  const [nerMethod, setNerMethod] = useState<NerMethod>("transformers");
  const [pipeBatchSize, setPipeBatchSize] = useState(4);
  const [pipeUseWikipedia, setPipeUseWikipedia] = useState(true);
  const [relevanceThreshold, setRelevanceThreshold] = useState(0.5);
  const [pipeResult, setPipeResult] = useState<Record<string, unknown> | null>(
    null
  );
  const [pipeCompareMethods, setPipeCompareMethods] = useState<boolean>(false);
  const [pipeMethodResults, setPipeMethodResults] = useState<
    Record<string, Record<string, unknown> | null>
  >({});
  const [pipeOptionalNotice, setPipeOptionalNotice] = useState<string | null>(null);
  const [normJson, setNormJson] = useState(
    `[{"pmid":"1","mention":"TGFB1","start":0,"end":5}]`
  );
  const [lastRunInfo, setLastRunInfo] = useState<LastRunInfo | null>(null);
  const trainFileRef = useRef<HTMLInputElement>(null);
  const pipeFileRef = useRef<HTMLInputElement>(null);
  const mentionFileRef = useRef<HTMLInputElement>(null);
  const litsuggestFileRef = useRef<HTMLInputElement>(null);

  const [pubmedEmail, setPubmedEmail] = useState("");
  const [pubmedQuery, setPubmedQuery] = useState(
    '("diabetic kidney disease"[Title/Abstract]) AND 2020:2024[dp]'
  );
  const [pubmedMax, setPubmedMax] = useState(50);
  const [pubmedMinAbstract, setPubmedMinAbstract] = useState(0);
  const [pubmedResult, setPubmedResult] = useState<{
    articles: { pmid: string; title: string; text: string }[];
    row_count: number;
    queried_id_count: number;
    search_total_estimate: number | null;
  } | null>(null);
  const [pubmedBusy, setPubmedBusy] = useState(false);

  const [comparePrimaryJson, setComparePrimaryJson] = useState(exampleComparePrimary);
  const [compareLitJson, setCompareLitJson] = useState(exampleCompareLitSuggest);
  const [compareThreshold, setCompareThreshold] = useState(0.5);
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null);
  const [compareBusy, setCompareBusy] = useState(false);
  const [baseModelCheck, setBaseModelCheck] = useState<ModelCheckState>({
    modelId: "",
    checking: false,
    compatible: null,
    detectedTasks: [],
    message: "Not checked",
  });
  const [nerModelCheck, setNerModelCheck] = useState<ModelCheckState>({
    modelId: "",
    checking: false,
    compatible: null,
    detectedTasks: [],
    message: "Not checked",
  });

  const isBackendConnected = backendStatus === "connected";
  const workspaceDraftSaveTimer = useRef<number | null>(null);
  const workspaceStateRestoredRef = useRef(false);
  const rememberRunningJob = (nextJobId: string | null) => {
    if (!projectId) return;
    const key = makeProjectJobStateKey(projectId);
    if (nextJobId) {
      localStorage.setItem(key, nextJobId);
    } else {
      localStorage.removeItem(key);
    }
    setJobId(nextJobId);
  };

  const refreshJobHistory = async (): Promise<JobRecord[] | null> => {
    if (!projectId || !isBackendConnected) return null;
    try {
      const recent = await api.listJobs(projectId, 20);
      setJobHistory(recent);
      return recent;
    } catch {
      setJobHistory([]);
      return null;
    }
  };

  const restoreWorkspaceDraft = () => {
    if (!projectId) return;
    const key = makeProjectWorkspaceStateKey(projectId);
    const raw = localStorage.getItem(key);
    const fallbackCfg = defaultTrainConfig();
    workspaceStateRestoredRef.current = false;

    if (!raw) {
      workspaceStateRestoredRef.current = true;
      return;
    }

    try {
      const parsed = JSON.parse(raw) as Partial<PersistedWorkspaceState>;
      if (typeof parsed.version === "number" && parsed.version >= 1) {
        if (typeof parsed.articlesJson === "string") setArticlesJson(parsed.articlesJson);
        if (typeof parsed.pipeJson === "string") setPipeJson(parsed.pipeJson);
        if (typeof parsed.normJson === "string") setNormJson(parsed.normJson);
        if (typeof parsed.comparePrimaryJson === "string") setComparePrimaryJson(parsed.comparePrimaryJson);
        if (typeof parsed.compareLitJson === "string") setCompareLitJson(parsed.compareLitJson);
        if (typeof parsed.pubmedEmail === "string") setPubmedEmail(parsed.pubmedEmail);
        if (typeof parsed.pubmedQuery === "string") setPubmedQuery(parsed.pubmedQuery);

        const restoredTrainCfg = typeof parsed.trainCfg === "object" && parsed.trainCfg !== null
          ? (() => {
              const rawTrainCfg = parsed.trainCfg as StoredTrainingConfig;
              const current: StoredTrainingConfig = {
                ...rawTrainCfg,
              };
              return {
              ...fallbackCfg,
              ...current,
              processor: isProcessor(current.processor)
                ? (current.processor as Processor)
                : fallbackCfg.processor,
              fp16:
                current.fp16 === null || typeof current.fp16 === "boolean"
                  ? (current.fp16 as null | boolean)
                  : fallbackCfg.fp16,
              } as TrainingConfig;
            })()
          : fallbackCfg;
        setTrainCfg(restoredTrainCfg);

        setValSplit(toStoredNumberFromString(parsed.valSplit, 0.2));
        setKfoldSplits(Math.max(2, Math.round(toStoredNumber(parsed.kfoldSplits, 5))));
        setModelId(toStoredString(parsed.modelId, ""));
        setMode(isPipelineMode(parsed.mode) ? parsed.mode : "full");
        setProcessor(isProcessor(parsed.processor) ? parsed.processor : "auto");
        setNerModel(toStoredString(parsed.nerModel, "pruas/BENT-PubMedBERT-NER-Gene"));
        setNerMethod(isNerMethod(parsed.nerMethod) ? parsed.nerMethod : "transformers");
        setBentServiceUrl(toStoredString(parsed.bentServiceUrl, ""));
        setPipeBatchSize(Math.max(1, Math.round(toStoredNumber(parsed.pipeBatchSize, 4))));
        setPipeUseWikipedia(toStoredBoolean(parsed.pipeUseWikipedia, true));
        setRelevanceThreshold(
          toStoredNumberFromString(parsed.relevanceThreshold, 0.5)
        );
        setPipeCompareMethods(toStoredBoolean(parsed.pipeCompareMethods, false));
        setBaseModelToDownload(toStoredString(parsed.baseModelToDownload, ""));
        setCompareThreshold(toStoredNumberFromString(parsed.compareThreshold, 0.5));
        setPubmedMax(Math.max(1, Math.round(toStoredNumber(parsed.pubmedMax, 50))));
        setPubmedMinAbstract(Math.max(0, Math.round(toStoredNumber(parsed.pubmedMinAbstract, 0))));
        setActiveStep(isWorkspaceStep(parsed.activeStep) ? parsed.activeStep : "backend");
      }
    } catch {
      // keep defaults on parse issues
    }
    workspaceStateRestoredRef.current = true;
  };

  const resetWorkspaceDraft = () => {
    if (!projectId) return;
    if (!window.confirm("Reset workspace draft for this project?")) {
      return;
    }
    const draftKey = makeProjectWorkspaceStateKey(projectId);
    localStorage.removeItem(draftKey);
    setArticlesJson(exampleArticlesJson);
    setPipeJson(examplePipelineArticlesJson);
    setNormJson(`[{"pmid":"1","mention":"TGFB1","start":0,"end":5}]`);
    setComparePrimaryJson(exampleComparePrimary);
    setCompareLitJson(exampleCompareLitSuggest);
    setPubmedEmail("");
    setPubmedQuery('("diabetic kidney disease"[Title/Abstract]) AND 2020:2024[dp]');
    setTrainCfg(defaultTrainConfig());
    setValSplit(0.2);
    setKfoldSplits(5);
    setModelId("");
    setMode("full");
    setProcessor("auto");
    setNerModel("pruas/BENT-PubMedBERT-NER-Gene");
    setNerMethod("transformers");
    setBentServiceUrl("");
    setPipeBatchSize(4);
    setPipeUseWikipedia(true);
    setPipeCompareMethods(false);
    setRelevanceThreshold(0.5);
    setBaseModelToDownload("");
    setCompareThreshold(0.5);
    setPubmedMax(50);
    setPubmedMinAbstract(0);
    setActiveStep("backend");
    setImportNotice(null);
    setCompareResult(null);
    setPubmedResult(null);
  };

  const saveWorkspaceDraft = () => {
    if (!projectId || !workspaceStateRestoredRef.current) return;
    if (workspaceDraftSaveTimer.current) {
      window.clearTimeout(workspaceDraftSaveTimer.current);
    }
    workspaceDraftSaveTimer.current = window.setTimeout(() => {
      const payload: PersistedWorkspaceState = {
        version: WORKSPACE_STATE_VERSION,
        articlesJson,
        pipeJson,
        normJson,
        trainCfg,
        valSplit,
        kfoldSplits,
        modelId,
        mode,
        processor,
        nerModel,
        nerMethod,
        bentServiceUrl,
        pipeCompareMethods,
        pipeBatchSize,
        pipeUseWikipedia,
        relevanceThreshold,
        baseModelToDownload,
        comparePrimaryJson,
        compareLitJson,
        compareThreshold,
        pubmedEmail,
        pubmedQuery,
        pubmedMax,
        pubmedMinAbstract,
        activeStep,
      };
      localStorage.setItem(makeProjectWorkspaceStateKey(projectId), JSON.stringify(payload));
    }, 350);
  };

  const checkBaseModelCompatibility = async (
    modelId = trainCfg.base_model,
    force = false
  ): Promise<ModelCompatibilityResult | null> => {
    const modelIdTrimmed = modelId.trim();
    if (!isBackendConnected) {
      setBaseModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: null,
        detectedTasks: [],
        message: "Backend not connected; validation unavailable.",
      });
      return null;
    }
    if (!modelIdTrimmed) {
      setBaseModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: false,
        detectedTasks: [],
        message: "Base model is required.",
      });
      return null;
    }

    if (!force && baseModelCheck.modelId === modelIdTrimmed && baseModelCheck.compatible !== null) {
      return {
        model_id: baseModelCheck.modelId,
        expected_task: "classification",
        compatible: baseModelCheck.compatible,
        detected_tasks: baseModelCheck.detectedTasks,
        message: baseModelCheck.message,
      };
    }

    setBaseModelCheck((prev) => ({
      ...prev,
      modelId: modelIdTrimmed,
      checking: true,
      message: `Checking ${modelIdTrimmed} for classification compatibility...`,
    }));
    try {
      const check = await api.validateModelTask(modelIdTrimmed, "classification");
      setBaseModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: check.compatible,
        detectedTasks: check.detected_tasks,
        message: check.message,
      });
      return check;
    } catch (err) {
      const message = `Validation failed: ${(err as Error).message}`;
      setBaseModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: false,
        detectedTasks: [],
        message,
      });
      return null;
    }
  };

  const checkNerModelCompatibility = async (
    modelId = nerModel,
    force = false
  ): Promise<ModelCompatibilityResult | null> => {
    const modelIdTrimmed = modelId.trim();
    if (!isBackendConnected) {
      setNerModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: null,
        detectedTasks: [],
        message: "Backend not connected; validation unavailable.",
      });
      return null;
    }
    if (!modelIdTrimmed) {
      setNerModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: false,
        detectedTasks: [],
        message: "NER model is required.",
      });
      return null;
    }

    if (!force && nerModelCheck.modelId === modelIdTrimmed && nerModelCheck.compatible !== null) {
      return {
        model_id: nerModelCheck.modelId,
        expected_task: "token_classification",
        compatible: nerModelCheck.compatible,
        detected_tasks: nerModelCheck.detectedTasks,
        message: nerModelCheck.message,
      };
    }

    setNerModelCheck((prev) => ({
      ...prev,
      modelId: modelIdTrimmed,
      checking: true,
      message: `Checking ${modelIdTrimmed} for token-classification compatibility...`,
    }));
    try {
      const check = await api.validateModelTask(modelIdTrimmed, "token_classification");
      setNerModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: check.compatible,
        detectedTasks: check.detected_tasks,
        message: check.message,
      });
      return check;
    } catch (err) {
      const message = `Validation failed: ${(err as Error).message}`;
      setNerModelCheck({
        modelId: modelIdTrimmed,
        checking: false,
        compatible: false,
        detectedTasks: [],
        message,
      });
      return null;
    }
  };

  const validateBaseModelBeforeTrain = async (): Promise<boolean> => {
    const check = await checkBaseModelCompatibility(trainCfg.base_model, true);
    if (!check) {
      setErr("Base model check failed.");
      return false;
    }
    if (!check.compatible) {
      setErr(`Base model check failed: ${check.message}`);
      return false;
    }
    return true;
  };

  const validateNerModelForPipeline = async (method: NerMethod): Promise<boolean> => {
    if (method !== "transformers") {
      return true;
    }
    const check = await checkNerModelCompatibility(nerModel, true);
    if (!check) {
      setErr("NER model check failed.");
      return false;
    }
    if (!check.compatible) {
      setErr(`NER model check failed: ${check.message}`);
      return false;
    }
    return true;
  };

  const restoreRunningJob = async () => {
    if (!projectId || !isBackendConnected) return;
    const key = makeProjectJobStateKey(projectId);
    try {
      const recent = await api.listJobs(projectId, 20);
      setJobHistory(recent);
      const runningJob = recent.find((j) => isJobStateRunning(j.state));
      const completedJobWithResult = recent.find(
        (j) => isJobStateTerminal(j.state) && j.result !== null && j.result !== undefined
      );
      const jobToCheck = runningJob?.job_id || localStorage.getItem(key);
      if (jobToCheck) {
        try {
          const status = await api.jobStatus(jobToCheck);
          setJobPoll(status);
          setJobId(status.job_id);
          if (!isJobStateRunning((status as { state?: unknown }).state)) {
            rememberRunningJob(null);
          }
          return;
        } catch {
          localStorage.removeItem(key);
          rememberRunningJob(null);
        }
      }
      if (!completedJobWithResult) return;
      setJobPoll(completedJobWithResult);
      setJobId(completedJobWithResult.job_id);
      rememberRunningJob(null);
    } catch {
      setJobHistory([]);
      localStorage.removeItem(key);
      setJobPoll(null);
      rememberRunningJob(null);
    }
  };

  const renderModelCheckTag = (label: string, check: ModelCheckState) => {
    if (check.checking) {
      return (
        <div className="tag" style={{ marginTop: "0.35rem" }}>
          {label}: checking...
        </div>
      );
    }

    if (check.compatible === null) {
      return (
        <div className="tag" style={{ marginTop: "0.35rem", opacity: 0.8 }}>
          {label}: not checked
        </div>
      );
    }

    if (check.compatible) {
      return (
        <div
          className="tag"
          style={{ marginTop: "0.35rem", color: "var(--success)" }}
        >
          {label}: compatible
        </div>
      );
    }

    return (
      <div
        className="tag"
        style={{ marginTop: "0.35rem", color: "var(--danger)" }}
      >
        {label}: incompatible
      </div>
    );
  };
  const jobState = (() => {
    if (typeof jobPoll !== "object" || jobPoll === null) return "";
    if (!("state" in jobPoll)) return "";
    const rawState = (jobPoll as { state?: unknown }).state;
    return typeof rawState === "string" ? rawState.toLowerCase() : "";
  })();

  const jobProgressPercent = (() => {
    if (typeof jobPoll !== "object" || jobPoll === null) return null;
    if (!("progress" in jobPoll)) return null;
    return toPercentOrNull((jobPoll as { progress?: unknown }).progress);
  })();

  const isJobRunning = Boolean(
    isJobStateRunning(jobState) &&
      (typeof jobPoll === "object" && jobPoll !== null)
  );

  const jobRemainingSeconds = (() => {
    if (!isJobRunning || jobProgressPercent === null) return null;
    if (jobProgressPercent <= 0.1) return null;
    const createdAt = (() => {
      if (typeof jobPoll !== "object" || jobPoll === null) return null;
      if (!("created_at" in jobPoll)) return null;
      return toEpochMsOrNull((jobPoll as { created_at?: unknown }).created_at);
    })();
    if (createdAt === null) return null;
    const elapsedSec = Math.max(0, (Date.now() - createdAt) / 1000);
    if (elapsedSec <= 0) return null;
    const progressRatio = jobProgressPercent / 100;
    if (!progressRatio || !Number.isFinite(progressRatio)) return null;
    const estimatedTotalSec = elapsedSec / progressRatio;
    return Math.max(0, Math.round(estimatedTotalSec - elapsedSec));
  })();
  const renderModelCheckMessage = (check: ModelCheckState) => {
    if (!check.message) return null;
    const color =
      check.compatible === null
        ? "var(--muted)"
        : check.compatible
          ? "var(--success)"
          : "var(--danger)";
    return (
      <p
        className="mono"
        style={{ margin: "0.25rem 0 0", color, fontSize: "0.78rem", maxWidth: "100%" }}
      >
        {check.message}
      </p>
    );
  };

  const jumpToStep = (step: WorkspaceStep) => {
    setActiveStep(step);
    const node = document.getElementById(`workspace-${step}`);
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const refreshLastRun = () => {
    if (!projectId) return;
    api
      .lastRun(projectId)
      .then(setLastRunInfo)
      .catch(() => setLastRunInfo(null));
  };

  const refreshModels = () => {
    if (!projectId) return;
    api
      .listProjectModelCatalog(projectId)
      .then((r) => {
        setProjectModels(r.models);
        const names = r.models.map((m) => m.model_id);
        setModels(names);
        if (names.length && !modelId) {
          setModelId(names[names.length - 1]!);
        }
      })
      .catch(() => {
        api.listModels(projectId).then((r) => {
          const names = r.models;
          setModels(names);
          setProjectModels(names.map((model_id) => ({ model_id, path: "" })));
          if (names.length && !modelId) {
            setModelId(names[names.length - 1]!);
          }
        }).catch(() => {
          setModels([]);
          setProjectModels([]);
        });
      });
  };

  const refreshBaseModels = async () => {
    try {
      const result = await api.listBaseModels();
      setAvailableBaseModels(result.models);
    } catch {
      setAvailableBaseModels([]);
    }
  };

  useEffect(() => {
    api.devices().then((d) => {
      setDevices(
        `Recommended: ${d.recommended} · CUDA ${d.available.cuda ? "on" : "off"
        } · MPS ${d.available.mps ? "on" : "off"}`
      );
    });
  }, []);

  useEffect(() => {
    refreshModels();
  }, [projectId]);

  useEffect(() => {
    refreshLastRun();
  }, [projectId]);

  useEffect(() => {
    restoreWorkspaceDraft();
  }, [projectId]);

  useEffect(() => {
    if (!projectId || !workspaceStateRestoredRef.current) return;
    saveWorkspaceDraft();
    return () => {
      if (workspaceDraftSaveTimer.current) {
        window.clearTimeout(workspaceDraftSaveTimer.current);
        workspaceDraftSaveTimer.current = null;
      }
    };
  }, [
    projectId,
    articlesJson,
    pipeJson,
    normJson,
    trainCfg,
    valSplit,
    kfoldSplits,
    modelId,
    mode,
    processor,
    nerModel,
    pipeBatchSize,
    pipeUseWikipedia,
    relevanceThreshold,
    baseModelToDownload,
    comparePrimaryJson,
    compareLitJson,
    compareThreshold,
    pubmedEmail,
    pubmedQuery,
    pubmedMax,
    pubmedMinAbstract,
    activeStep,
  ]);

  useEffect(() => {
    if (!isBackendConnected) return;
    void restoreRunningJob();
    void refreshJobHistory();
  }, [isBackendConnected, projectId]);

  useEffect(() => {
    if (!isBackendConnected) return;
    const handle = window.setTimeout(() => {
      void checkBaseModelCompatibility(trainCfg.base_model);
    }, 450);
    return () => clearTimeout(handle);
  }, [trainCfg.base_model, isBackendConnected]);

  useEffect(() => {
    if (!isBackendConnected) return;
    const handle = window.setTimeout(() => {
      void checkNerModelCompatibility(nerModel);
    }, 450);
    return () => clearTimeout(handle);
  }, [nerModel, isBackendConnected]);

  useEffect(() => {
    if (!jobId || !isJobStateRunning(jobState)) return;
    const t = setInterval(() => {
      api
        .jobStatus(jobId)
        .then((j) => {
          setJobPoll(j);
          if (j.state === "completed" || j.state === "failed") {
            clearInterval(t);
            refreshModels();
            void refreshJobHistory();
            rememberRunningJob(null);
          }
        })
        .catch(() => {
          clearInterval(t);
          void refreshJobHistory();
          rememberRunningJob(null);
          setJobPoll(null);
        });
    }, 1500);
    return () => clearInterval(t);
  }, [jobId, jobState]);

  const parseArticles = (): Article[] => {
    const raw = JSON.parse(articlesJson) as Article[];
    if (!Array.isArray(raw)) throw new Error("Expected JSON array");
    return raw.map((a) => {
      const titleStr =
        a.title !== undefined && a.title !== null && String(a.title).trim() !== ""
          ? String(a.title).trim()
          : undefined;
      return {
        pmid: String(a.pmid),
        text: String(a.text),
        ...(titleStr !== undefined ? { title: titleStr } : {}),
        label:
          a.label === undefined || a.label === null ? undefined : Number(a.label),
      };
    });
  };

  const jobRemainingText =
    jobRemainingSeconds === null ? null : formatSecondsAsHuman(jobRemainingSeconds);

  const jobMessage = (() => {
    if (typeof jobPoll !== "object" || jobPoll === null) return "";
    if (!("message" in jobPoll)) return "";
    const message = (jobPoll as { message?: unknown }).message;
    return typeof message === "string" ? message : "";
  })();

  const globalLoadingMessage = (() => {
    if (backendStatus === "checking") return "Checking backend...";
    if (backendDevicesStatus === "checking") return "Running backend system check...";
    if (isDownloadingBaseModel) return "Downloading base model...";
    if (pubmedBusy) return "Fetching PubMed data...";
    if (compareBusy) return "Running LitSuggest comparison...";
    if (busy && busyMessage) return busyMessage;
    if (isJobRunning) {
      return `Training job ${jobId}: ${jobStateText(jobState)}`;
    }
    if (busy) return "Working...";
    return null;
  })();

  const globalLoadingPercent = isJobRunning ? jobProgressPercent : null;

  const train = async (kfold: boolean) => {
    setErr(null);
    setBusy(true);
    setBusyMessage("Validating base model...");
    try {
      if (!(await validateBaseModelBeforeTrain())) {
        return;
      }
      setBusyMessage(kfold ? "Submitting k-fold training job..." : "Submitting training job...");
      const articles = parseArticles();
      const cfg = { ...trainCfg, n_splits: kfoldSplits };
      const res = kfold
        ? await api.trainKfold(projectId, articles, cfg)
        : await api.trainRelevance(projectId, articles, trainCfg, valSplit);
      const status = await api.jobStatus(res.job_id);
      setJobPoll(status);
      rememberRunningJob(res.job_id);
      void refreshJobHistory();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
      setBusyMessage(null);
    }
  };

  const runPipe = async () => {
    setErr(null);
    setPipeOptionalNotice(null);
    setBusy(true);
    setBusyMessage("Running pipeline jobs...");
    setPipeResult(null);
    setPipeMethodResults({});
    try {
      const compareEnabled = (mode === "ner" || mode === "full") && pipeCompareMethods;
      const methodsToRun: NerMethod[] = compareEnabled
        ? Array.from(new Set([nerMethod, "transformers", "bent"]))
        : [nerMethod];

      if (mode === "ner" || mode === "full") {
        setBusyMessage("Validating NER method compatibility...");
        for (const method of methodsToRun) {
          if (!(await validateNerModelForPipeline(method))) {
            return;
          }
        }
      }

      setBusyMessage("Running pipeline jobs...");
      const parsedArticles = JSON.parse(pipeJson) as Article[];
      const baseArticles = parsedArticles.map((a) => {
        const base = {
          pmid: String(a.pmid),
          text: String(a.text),
          label: a.label === undefined || a.label === null ? undefined : Number(a.label),
        };
        const titleStr =
          a.title !== undefined &&
          a.title !== null &&
          String(a.title).trim() !== ""
            ? String(a.title).trim()
            : undefined;
        return titleStr !== undefined ? { ...base, title: titleStr } : base;
      });

      const runSinglePipeline = async (method: NerMethod) => {
        const payload: Parameters<typeof api.runPipeline>[0] = {
          project_id: projectId,
          model_id: modelId,
          articles: baseArticles,
          mode,
          processor,
          ner_model: nerModel,
          ner_method: method,
          bent_service_url: bentServiceUrl.trim() || undefined,
          batch_size: pipeBatchSize,
          use_wikipedia_fallback: pipeUseWikipedia,
        };
        if (mode === "normalize") {
          payload.mentions_json = JSON.parse(normJson) as Record<string, unknown>[];
        }
        return api.runPipeline(payload);
      };

      const resultsByMethod: Record<string, Record<string, unknown> | null> = {};
      const failures: string[] = [];
      const skippedMethods: string[] = [];
      for (const method of methodsToRun) {
        const methodName = method === "transformers" ? "HF transformers" : "BENT";
        setBusyMessage(`Running pipeline jobs (${methodName})...`);
        try {
          const out = await runSinglePipeline(method);
          resultsByMethod[method] = out;
        } catch (err) {
          const message = (err as Error).message;
          if (method === "bent" && isBentUnavailableError(message)) {
            const reason = getErrorDetailMessage(message);
            if (compareEnabled) {
              skippedMethods.push(`BENT: ${reason}`);
              resultsByMethod[method] = null;
              continue;
            }
            failures.push(`BENT: ${reason}`);
            resultsByMethod[method] = null;
            continue;
          }
          failures.push(`${methodName}: ${message}`);
          resultsByMethod[method] = null;
        }
      }

      const completed = Object.values(resultsByMethod).some((value) => value !== null);
      if (skippedMethods.length > 0) {
        setPipeOptionalNotice(`Optional methods were skipped: ${skippedMethods.join(" | ")}`);
      } else {
        setPipeOptionalNotice(null);
      }
      if (!completed) {
        throw new Error(failures.length ? failures.join(" | ") : "No pipeline run produced a result.");
      }

      setPipeMethodResults(resultsByMethod);
      const firstResult = resultsByMethod[methodsToRun[0]] || null;
      if (firstResult) {
        setPipeResult(firstResult);
      } else {
        const fallback = Object.values(resultsByMethod).find((value) => value !== null);
        if (fallback) {
          setPipeResult(fallback);
        }
      }
      if (failures.length) {
        setErr(`Some methods failed: ${failures.join(" | ")}`);
      }
      refreshLastRun();
    } catch (e) {
      setPipeOptionalNotice(null);
      setErr((e as Error).message);
    } finally {
      setBusy(false);
      setBusyMessage(null);
    }
  };

  const runPubmedFetch = async () => {
    if (!projectId) return;
    if (!pubmedEmail.trim() || !pubmedEmail.includes("@")) {
      setErr("Enter a valid email (required by NCBI Entrez).");
      return;
    }
    setErr(null);
    setPubmedBusy(true);
    setPubmedResult(null);
    try {
      const r = await api.pubmedFetch(projectId, {
        email: pubmedEmail.trim(),
        query: pubmedQuery.trim() || undefined,
        max_results: pubmedMax,
        min_abstract_chars: pubmedMinAbstract,
      });
      setPubmedResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setPubmedBusy(false);
    }
  };

  const copyPubmedToPipelineJson = () => {
    if (!pubmedResult?.articles?.length) return;
    const lines = pubmedResult.articles.map((a) => {
      const row: Record<string, string> = {
        pmid: a.pmid,
        text: a.text || "",
      };
      if (a.title?.trim()) row.title = a.title.trim();
      return row;
    });
    setPipeJson(JSON.stringify(lines, null, 2));
    jumpToStep("pipeline");
  };

  const runLitCompare = async () => {
    setErr(null);
    setCompareBusy(true);
    setCompareResult(null);
    try {
      const primary = JSON.parse(comparePrimaryJson) as Record<string, unknown>[];
      const litsuggest = JSON.parse(compareLitJson) as Record<string, unknown>[];
      if (!Array.isArray(primary) || !Array.isArray(litsuggest)) {
        throw new Error("Expected two JSON arrays");
      }
      const r = await api.compareLitSuggest(projectId, {
        primary,
        litsuggest,
        score_threshold: compareThreshold,
      });
      setCompareResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setCompareBusy(false);
    }
  };

  const onLitSuggestFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setErr(null);
    setCompareBusy(true);
    try {
      const r = await api.importLitSuggestScores(projectId, f);
      setCompareLitJson(JSON.stringify(r.litsuggest, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setCompareBusy(false);
    }
  };

  const applyTrainPreset = () => {
    setTrainCfg(mpsTrainPreset());
    setValSplit(0.2);
    setKfoldSplits(5);
  };

  const onTrainFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setBusyMessage("Importing training file...");
    setErr(null);
    setImportNotice(null);
    try {
      const r = await api.importArticles(projectId, f);
      setArticlesJson(JSON.stringify(r.articles, null, 2));
      setImportNotice(formatImportNotice("Train import", r.import_stats));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
      setBusyMessage(null);
    }
  };

  const onPipeArticlesFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setBusyMessage("Importing pipeline file...");
    setErr(null);
    setImportNotice(null);
    try {
      const r = await api.importArticles(projectId, f);
      setPipeJson(JSON.stringify(r.articles, null, 2));
      setImportNotice(formatImportNotice("Pipeline import", r.import_stats));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
      setBusyMessage(null);
    }
  };

  const onMentionsFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f || !projectId) return;
    setBusy(true);
    setBusyMessage("Importing mentions file...");
    setErr(null);
    try {
      const r = await api.importMentions(projectId, f);
      setNormJson(JSON.stringify(r.mentions, null, 2));
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setBusy(false);
      setBusyMessage(null);
    }
  };

  const downloadBaseModel = async () => {
    const modelIdTrimmed = baseModelToDownload.trim();
    if (!modelIdTrimmed) {
      setErr("Enter a model id before downloading.");
      return;
    }
    if (availableBaseModels.includes(modelIdTrimmed)) {
      setTrainCfg((cfg) => ({ ...cfg, base_model: modelIdTrimmed }));
      setBaseModelToDownload("");
      return;
    }
    setIsDownloadingBaseModel(true);
    setErr(null);
    setBusyMessage(`Downloading model: ${modelIdTrimmed}`);
    try {
      const result = await api.downloadBaseModel(modelIdTrimmed);
      setTrainCfg((cfg) => ({ ...cfg, base_model: result.model_id }));
      setBaseModelToDownload("");
      await refreshBaseModels();
    } catch (err) {
      setErr((err as Error).message);
    } finally {
      setIsDownloadingBaseModel(false);
      setBusyMessage(null);
    }
  };

  const runBackendCheck = async (candidate?: string) => {
    setBackendStatus("checking");
    setBackendStatusText("Checking...");
    try {
      await pingBackend(candidate);
      if (candidate !== undefined) {
        setBackendUrl(setApiBase(candidate));
      }
      setBackendStatus("connected");
      setBackendStatusText("Connected");
      setBackendMessage(
        "Backend connected. Run system check, then start training or pipeline."
      );
      setErr(null);
      void refreshBaseModels();
      void refreshModels();
      void refreshLastRun();
      void restoreRunningJob();
      void refreshJobHistory();
    } catch (err) {
      setBackendStatus("failed");
      setBackendStatusText((err as Error).message || "Unable to connect");
      setBackendMessage("Save backend URL here and click save, then retry the check.");
      setBackendDevicesStatus("pending");
      setBackendDevices(null);
      setAvailableBaseModels([]);
      setJobHistory([]);
    }
  };

  const runSystemCheck = async () => {
    setBackendDevicesStatus("checking");
    try {
      const d = await api.devices();
      setBackendDevices(d);
      setBackendDevicesStatus("done");
    } catch (err) {
      setBackendDevicesStatus("error");
      setErr((err as Error).message);
      setBackendMessage((err as Error).message || "System check failed.");
    }
  };

  useEffect(() => {
    const sync = () => {
      const next = getApiBase();
      setBackendUrl(next);
      void runBackendCheck();
    };
    sync();
    const stop = apiBaseDidChange(sync);
    return stop;
  }, []);

  const applyPipelinePreset = () => {
    setProcessor("mps");
    setNerModel("pruas/BENT-PubMedBERT-NER-Gene");
    setNerMethod("transformers");
    setPipeBatchSize(4);
    setPipeUseWikipedia(true);
    setMode("full");
    setPipeCompareMethods(false);
    setRelevanceThreshold(0.5);
  };

  const jobHasResult = (result: JobRecord["result"]) => {
    if (result === null || result === undefined) return false;
    if (Array.isArray(result)) return result.length > 0;
    if (typeof result === "object") return Object.keys(result).length > 0;
    return true;
  };

  const classificationRows = pipeResult ? getClassificationRows(pipeResult) : [];
  const methodComparisonRows = Object.entries(pipeMethodResults).map(([method, result]) => ({
    method,
    rows: getMentionsRows(result),
  }));
  const filteredClassificationRows = classificationRows.filter((row) => {
    const prob = toNumberOrNull(row.relevance_prob);
    if (prob === null) return false;
    return prob >= relevanceThreshold;
  });
  const filteredJobsForHistory = jobHistory.filter((job) =>
    showOnlyJobsWithResult
      ? isJobStateTerminal(job.state) && jobHasResult(job.result)
      : true
  );
  const recentJobHistory = filteredJobsForHistory.slice(0, 10);
  const formatEpochLabel = (value: string) => {
    const parsed = Date.parse(value);
    if (Number.isNaN(parsed)) return value;
    return new Date(parsed).toLocaleString();
  };
  const pickJob = (job: JobRecord) => {
    setJobPoll(job);
    setJobId(job.job_id);
    if (!isJobStateRunning(job.state)) {
      rememberRunningJob(null);
    } else {
      rememberRunningJob(job.job_id);
    }
  };

  return (
    <div>
      <p>
        <Link to="/projects">← Projects</Link>
      </p>
      <h1>Project workspace</h1>
      <div className="steps">
        <a
          href="#workspace-backend"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("backend");
          }}
          className={activeStep === "backend" ? "step-pill active" : "step-pill"}
        >
          1 · Backend
        </a>
        <a
          href="#workspace-train"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("train");
          }}
          className={activeStep === "train" ? "step-pill active" : "step-pill"}
        >
          2 · Train
        </a>
        <a
          href="#workspace-integrations"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("integrations");
          }}
          className={activeStep === "integrations" ? "step-pill active" : "step-pill"}
        >
          3 · PubMed & LitSuggest
        </a>
        <a
          href="#workspace-pipeline"
          onClick={(e) => {
            e.preventDefault();
            jumpToStep("pipeline");
          }}
          className={activeStep === "pipeline" ? "step-pill active" : "step-pill"}
        >
          4 · Pipeline
        </a>
      </div>
      <div className="toolbar" style={{ marginBottom: "1rem" }}>
        <Link to={`/jobs${projectId ? `?projectId=${projectId}` : ""}`} className="btn btn-primary">
          Open job center
        </Link>
        <button
          type="button"
          className="btn"
          onClick={resetWorkspaceDraft}
          disabled={busy}
        >
          Reset workspace draft
        </button>
        {!isBackendConnected && (
          <p style={{ margin: 0, color: "var(--warning)" }}>
            Backend is not connected. Connect first before training/pipeline.
          </p>
        )}
      </div>
      <div id="workspace-backend" className="card">
        <h3 style={{ marginTop: 0 }}>Backend status</h3>
        <p style={{ marginTop: 0, color: "var(--muted)" }}>{backendMessage}</p>
        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
          URL: <code className="mono">{backendUrl}</code> · Status:{" "}
          <strong style={{ color: backendStatus === "connected" ? "var(--success)" : backendStatus === "failed" ? "var(--danger)" : "inherit" }}>
            {backendStatusText}
          </strong>
        </p>
        {backendDevices && (
          <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
            Recommended: {backendDevices.recommended.toUpperCase()} · CUDA:{" "}
            {backendDevices.available.cuda ? "on" : "off"} · MPS:{" "}
            {backendDevices.available.mps ? "on" : "off"} · CPU:{" "}
            {backendDevices.available.cpu ? "on" : "off"}
          </p>
        )}
        <div className="toolbar" style={{ alignItems: "flex-start" }}>
          <input
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            style={{ maxWidth: "540px", flex: 1 }}
            placeholder="/api or https://....ngrok-free.app/api"
          />
          <button
            className="btn"
            type="button"
            onClick={() => runBackendCheck(backendUrl)}
            disabled={backendStatus === "checking"}
          >
            Save & test
          </button>
          <button
            className="btn"
            type="button"
            onClick={runSystemCheck}
            disabled={backendStatus !== "connected" || backendDevicesStatus === "checking"}
          >
            Check system
          </button>
        </div>
        <div style={{ marginTop: "0.75rem", display: "grid", gap: "0.35rem" }}>
          <label style={{ fontSize: "0.9rem" }}>Bent service endpoint (optional)</label>
          <input
            value={bentServiceUrl}
            onChange={(e) => setBentServiceUrl(e.target.value)}
            style={{ maxWidth: "540px" }}
            placeholder="http://bent-runtime:8010 or http://<host>:8010"
          />
          <p style={{ margin: 0, color: "var(--muted)", fontSize: "0.8rem" }}>
            Leave empty to fall back to backend env (`BENT_SERVICE_URL`). Add ` /annotate` only if needed for direct calls.
          </p>
        </div>
        <div style={{ fontSize: "0.8rem", marginTop: "0.5rem", color: "var(--muted)" }}>
          {backendDevicesStatus === "checking" ? "Checking devices..." : backendDevicesStatus === "done" ? "System check done." : ""}
        </div>
      </div>
      <p className="mono" style={{ color: "var(--muted)" }}>
        {projectId}
      </p>
      {devices && (
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>{devices}</p>
      )}

      {globalLoadingMessage && (
        <div className="card" style={{ borderColor: "var(--accent)" }}>
          <div className="loading-inline" role="status" aria-live="polite">
            <span className="loading-spinner" aria-hidden="true" />
            <span className="mono">
              {globalLoadingMessage}
              {isJobRunning && globalLoadingPercent !== null
                ? ` · ${Math.round(globalLoadingPercent)}%`
                : ""}
              {isJobRunning && jobRemainingText ? ` · ETA ${jobRemainingText}` : ""}
            </span>
          </div>
          <div className="loading-track">
            <div
              className={`loading-fill ${
                globalLoadingPercent !== null
                  ? "loading-fill-determinate"
                  : "loading-fill-indeterminate"
              }`}
              style={
                globalLoadingPercent !== null
                  ? { width: `${Math.max(4, globalLoadingPercent)}%` }
                  : undefined
              }
            />
          </div>
        </div>
      )}

      {err && (
        <div className="card" style={{ borderColor: "var(--danger)" }}>
          {err}
        </div>
      )}
      {pipeOptionalNotice && (
        <div className="card" style={{ borderColor: "var(--warning)" }}>
          {pipeOptionalNotice}
        </div>
      )}
      {importNotice && (
        <div
          className="card"
          style={{
            borderColor:
              importNotice.includes("skipped:") || importNotice.includes("no valid")
                ? "var(--warning)"
                : "var(--success)",
          }}
        >
          {importNotice}
        </div>
      )}

      <div id="workspace-train" className="card">
        <h3 style={{ marginTop: 0 }}>2 · Train relevance (BioBERT)</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Paste JSON array: <code className="mono">pmid</code>,{" "}
          <code className="mono">text</code> (abstract or body),{" "}
          <code className="mono">label</code> (0 or 1). Optional{" "}
          <code className="mono">title</code> enables title+abstract pair encoding (DKDM-style).
          Imports deduplicate by PMID (first row kept).
        </p>
        <textarea
          value={articlesJson}
          onChange={(e) => setArticlesJson(e.target.value)}
          style={{ minHeight: "160px", fontFamily: "JetBrains Mono, monospace" }}
        />
        <input
          ref={trainFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onTrainFile}
        />
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => trainFileRef.current?.click()}
          >
            Import articles (CSV / Excel / PKL)
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() =>
              api
                .downloadTemplate(projectId, "articles")
                .catch((err) => setErr((err as Error).message))
            }
          >
            Download template (CSV)
          </button>
        </div>

        <div className="grid2" style={{ marginTop: "1rem" }}>
          <div>
            <label>Processor</label>
            <select
              value={trainCfg.processor}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  processor: e.target.value as Processor,
                })
              }
            >
              <option value="auto">auto (CUDA → MPS → CPU)</option>
              <option value="cuda">CUDA (NVIDIA GPU)</option>
              <option value="mps">MPS (Apple Metal)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div>
            <label>Base model</label>
            <div className="toolbar" style={{ marginTop: "0.15rem" }}>
              <select
                value={trainCfg.base_model}
                onChange={(e) => {
                  const value = e.target.value;
                  setTrainCfg({ ...trainCfg, base_model: value });
                  void checkBaseModelCompatibility(value, true);
                }}
              >
                {availableBaseModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
                {!availableBaseModels.includes(trainCfg.base_model) && (
                  <option value={trainCfg.base_model}>{`${trainCfg.base_model} (custom)`}</option>
                )}
              </select>
              <button
                type="button"
                className="btn"
                disabled={busy || !isBackendConnected}
                onClick={() => void refreshBaseModels()}
              >
                Refresh
              </button>
            </div>
            <div>{renderModelCheckTag("Base model", baseModelCheck)}</div>
            {renderModelCheckMessage(baseModelCheck)}
            <div className="toolbar" style={{ marginTop: "0.6rem" }}>
              <input
                value={baseModelToDownload}
                onChange={(e) => setBaseModelToDownload(e.target.value)}
                placeholder="Type HF model id, e.g. dmis-lab/biobert-v1.1"
              />
              <button
                type="button"
                className="btn"
                disabled={!isBackendConnected || isDownloadingBaseModel || !baseModelToDownload.trim()}
                onClick={() => void downloadBaseModel()}
              >
                {isDownloadingBaseModel ? "Downloading..." : "Download model"}
              </button>
            </div>
          </div>
          <div>
            <label>Learning rate</label>
            <input
              type="number"
              step="any"
              value={trainCfg.learning_rate}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  learning_rate: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div>
            <label>Epochs</label>
            <input
              type="number"
              value={trainCfg.num_train_epochs}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  num_train_epochs: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Train batch size</label>
            <input
              type="number"
              value={trainCfg.per_device_train_batch_size}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  per_device_train_batch_size: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Eval batch size</label>
            <input
              type="number"
              value={trainCfg.per_device_eval_batch_size}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  per_device_eval_batch_size: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>Weight decay</label>
            <input
              type="number"
              step="any"
              value={trainCfg.weight_decay}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  weight_decay: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div>
            <label>Max sequence length</label>
            <input
              type="number"
              value={trainCfg.max_length}
              onChange={(e) =>
                setTrainCfg({
                  ...trainCfg,
                  max_length: parseInt(e.target.value, 10),
                })
              }
            />
          </div>
          <div>
            <label>FP16 (null = auto: on for CUDA only)</label>
            <select
              value={
                trainCfg.fp16 === null ? "auto" : trainCfg.fp16 ? "on" : "off"
              }
              onChange={(e) => {
                const v = e.target.value;
                setTrainCfg({
                  ...trainCfg,
                  fp16: v === "auto" ? null : v === "on",
                });
              }}
            >
              <option value="auto">auto</option>
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </div>
          <div>
            <label>Validation fraction (single train)</label>
            <input
              type="number"
              step="0.05"
              min={0.1}
              max={0.4}
              value={valSplit}
              onChange={(e) => setValSplit(parseFloat(e.target.value))}
            />
          </div>
          <div>
            <label>K-fold splits (k-fold only)</label>
            <input
              type="number"
              min={2}
              max={10}
              value={kfoldSplits}
              onChange={(e) => setKfoldSplits(parseInt(e.target.value, 10))}
            />
          </div>
        </div>

        <div className="toolbar" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="btn"
            disabled={!isBackendConnected || busy}
            onClick={applyTrainPreset}
          >
            Apply MPS safe preset
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || busy}
            onClick={() => train(false)}
          >
            Start training (holdout)
          </button>
          <button
            type="button"
            className="btn"
            disabled={!isBackendConnected || busy}
            onClick={() => train(true)}
          >
            Start k-fold CV
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("integrations")}
          >
            Next: PubMed & LitSuggest
          </button>
        </div>

        {jobPoll && (
          <div style={{ marginTop: "1rem" }}>
            <span className="tag">job {jobId}</span>
            {jobMessage && <div className="tag" style={{ marginLeft: "0.5rem" }}>{jobMessage}</div>}
            <div
              style={{
                marginTop: "0.5rem",
                display: "flex",
                flexWrap: "wrap",
                gap: "0.4rem",
                alignItems: "center",
              }}
            >
              <span className="tag">
                state: {jobState || "unknown"} · {isJobRunning ? "running" : "not running"}
              </span>
              <span className="tag">
                progress:{" "}
                {jobProgressPercent === null ? "estimating" : `${Math.round(jobProgressPercent)}%`}
              </span>
              <span className="tag">
                eta: {jobRemainingText || "estimating"}
              </span>
            </div>
            <div className="loading-track" style={{ marginTop: "0.65rem" }}>
              <div
                className={`loading-fill ${
                  jobProgressPercent !== null
                    ? "loading-fill-determinate"
                    : "loading-fill-indeterminate"
                }`}
                style={
                  jobProgressPercent !== null
                    ? { width: `${Math.max(4, jobProgressPercent)}%` }
                    : undefined
                }
              />
            </div>
            <pre
              className="mono"
              style={{
                marginTop: "0.5rem",
                padding: "0.75rem",
                background: "var(--bg)",
                borderRadius: 8,
                overflow: "auto",
                fontSize: "0.82rem",
              }}
            >
              {JSON.stringify(jobPoll, null, 2)}
            </pre>
          </div>
        )}

        {jobHistory.length > 0 && (
          <div style={{ marginTop: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
              <h4 style={{ margin: 0 }}>Recent jobs</h4>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  color: "var(--muted)",
                  fontSize: "0.84rem",
                }}
              >
                <input
                  type="checkbox"
                  checked={showOnlyJobsWithResult}
                  onChange={(event) => setShowOnlyJobsWithResult(event.target.checked)}
                />
                show only completed jobs with result
              </label>
            </div>
            {recentJobHistory.length === 0 ? (
              <p style={{ color: "var(--muted)", margin: 0 }}>
                No jobs match the selected history filter.
              </p>
            ) : (
              <div style={{ display: "grid", gap: "0.5rem" }}>
                {recentJobHistory.map((job) => {
                  const isCurrent = job.job_id === jobId;
                  return (
                    <div
                      key={job.job_id}
                      className="card"
                      style={{
                        borderColor: isCurrent ? "var(--accent)" : "var(--card-border)",
                        padding: "0.55rem 0.65rem",
                      }}
                    >
                      <div
                        style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", flexWrap: "wrap" }}
                      >
                        <div>
                          <div className="mono">
                            {job.job_id.slice(0, 8)} · {job.state.toUpperCase()}
                          </div>
                          <div className="tag" style={{ marginTop: "0.35rem" }}>
                            {formatEpochLabel(job.created_at)}{" "}
                            {isCurrent ? "· current view" : ""}
                          </div>
                          {job.message ? (
                            <div className="tag" style={{ marginTop: "0.35rem" }}>
                              {job.message}
                            </div>
                          ) : null}
                          {job.result !== undefined ? (
                            <div className="tag" style={{ marginTop: "0.35rem", opacity: 0.85 }}>
                              result: {jobHasResult(job.result) ? "yes" : "no"}
                            </div>
                          ) : null}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                          <button
                            type="button"
                            className="btn"
                            onClick={() => pickJob(job)}
                          >
                            Show
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      <div id="workspace-integrations" className="card">
        <h3 style={{ marginTop: 0 }}>3 · PubMed (Entrez) fetch & LitSuggest comparison</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Fetch citations from NCBI (email required), then optionally compare your binary labels to LitSuggest
          scores (DKDM-style threshold on scores).
        </p>

        <h4 style={{ marginBottom: "0.5rem" }}>PubMed fetch</h4>
        <div className="grid2">
          <div>
            <label>Entrez email (required by NCBI)</label>
            <input
              value={pubmedEmail}
              onChange={(e) => setPubmedEmail(e.target.value)}
              placeholder="you@institution.edu"
              autoComplete="email"
              style={{ width: "100%" }}
            />
          </div>
          <div>
            <label>Max articles / PMIDs</label>
            <input
              type="number"
              min={1}
              max={10000}
              value={pubmedMax}
              onChange={(e) => setPubmedMax(Number(e.target.value) || 50)}
            />
          </div>
        </div>
        <label style={{ display: "block", marginTop: "0.75rem" }}>PubMed query (esearch)</label>
        <textarea
          value={pubmedQuery}
          onChange={(e) => setPubmedQuery(e.target.value)}
          style={{ minHeight: "72px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <div className="grid2" style={{ marginTop: "0.75rem" }}>
          <div>
            <label>Min abstract length (0 = off)</label>
            <input
              type="number"
              min={0}
              value={pubmedMinAbstract}
              onChange={(e) => setPubmedMinAbstract(Number(e.target.value) || 0)}
            />
            <p style={{ fontSize: "0.78rem", color: "var(--muted)", margin: "0.35rem 0 0" }}>
              Matches DKDM: skip very short abstracts when set (e.g. 200).
            </p>
          </div>
        </div>
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || pubmedBusy}
            onClick={() => void runPubmedFetch()}
          >
            {pubmedBusy ? "Fetching…" : "Fetch from PubMed"}
          </button>
          <button
            type="button"
            className="btn"
            disabled={!pubmedResult?.articles?.length}
            onClick={() => copyPubmedToPipelineJson()}
          >
            Send results → pipeline JSON
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("train")}
          >
            Back
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("pipeline")}
          >
            Next: pipeline
          </button>
        </div>
        {pubmedResult && (
          <pre
            className="mono"
            style={{
              marginTop: "0.75rem",
              padding: "0.75rem",
              background: "var(--bg)",
              borderRadius: 8,
              overflow: "auto",
              fontSize: "0.78rem",
              maxHeight: "240px",
            }}
          >
            {JSON.stringify(pubmedResult, null, 2)}
          </pre>
        )}

        <h4 style={{ marginTop: "1.5rem", marginBottom: "0.5rem" }}>LitSuggest comparison</h4>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Primary: your labels (e.g. from training export or <code className="mono">label</code> /{" "}
          <code className="mono">relevant</code>). Secondary: LitSuggest <code className="mono">pmid</code> +{" "}
          <code className="mono">score</code>. Import LitSuggest CSV or paste JSON.
        </p>
        <input
          ref={litsuggestFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl"
          style={{ display: "none" }}
          onChange={onLitSuggestFile}
        />
        <div className="toolbar" style={{ flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn"
            disabled={compareBusy}
            onClick={() => litsuggestFileRef.current?.click()}
          >
            Import LitSuggest file
          </button>
        </div>
        <label style={{ display: "block", marginTop: "0.75rem" }}>Primary labels (JSON array)</label>
        <textarea
          value={comparePrimaryJson}
          onChange={(e) => setComparePrimaryJson(e.target.value)}
          style={{ minHeight: "100px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <label style={{ display: "block", marginTop: "0.75rem" }}>LitSuggest scores (JSON array)</label>
        <textarea
          value={compareLitJson}
          onChange={(e) => setCompareLitJson(e.target.value)}
          style={{ minHeight: "100px", fontFamily: "JetBrains Mono, monospace", width: "100%" }}
        />
        <div style={{ marginTop: "0.75rem" }}>
          <label>Score threshold for binary label (≥ = relevant)</label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={compareThreshold}
            onChange={(e) => setCompareThreshold(Number(e.target.value))}
            style={{ width: "100%", maxWidth: "360px", verticalAlign: "middle" }}
          />{" "}
          <span className="mono">{compareThreshold.toFixed(2)}</span>
        </div>
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || compareBusy}
            onClick={() => void runLitCompare()}
          >
            {compareBusy ? "Comparing…" : "Run comparison"}
          </button>
        </div>
        {compareResult && (
          <pre
            className="mono"
            style={{
              marginTop: "0.75rem",
              padding: "0.75rem",
              background: "var(--bg)",
              borderRadius: 8,
              overflow: "auto",
              fontSize: "0.78rem",
              maxHeight: "320px",
            }}
          >
            {JSON.stringify(compareResult, null, 2)}
          </pre>
        )}
      </div>

      <div id="workspace-pipeline" className="card">
        <h3 style={{ marginTop: 0 }}>4 · Run pipeline</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Pick a trained <span className="mono">model_id</span>, choose a
          step or full workflow, then run on new abstracts (labels optional for
          inference).
        </p>
        <input
          ref={pipeFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onPipeArticlesFile}
        />
        <input
          ref={mentionFileRef}
          type="file"
          accept=".csv,.xlsx,.xls,.pkl,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          style={{ display: "none" }}
          onChange={onMentionsFile}
        />
        <div className="toolbar" style={{ marginTop: "0.5rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => pipeFileRef.current?.click()}
          >
            Import articles file
          </button>
          {mode === "normalize" && (
            <button
              type="button"
              className="btn"
              disabled={busy}
              onClick={() => mentionFileRef.current?.click()}
            >
              Import mentions file
            </button>
          )}
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() =>
              api
                .downloadTemplate(projectId, "mentions")
                .catch((err) => setErr((err as Error).message))
            }
          >
            Mentions template (CSV)
          </button>
        </div>
        <div className="grid2">
          <div>
            <label>Trained model</label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              <option value="">— select —</option>
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          {projectModels.length > 0 && (
            <div style={{ gridColumn: "1 / -1" }}>
              <label>Available trained model folders (absolute paths)</label>
              <div style={{ marginTop: "0.5rem", display: "grid", gap: "0.4rem" }}>
                {projectModels.map((m) => (
                  <div
                    key={m.model_id}
                    style={{
                      background: "var(--bg)",
                      borderRadius: 8,
                      border: "1px solid var(--border)",
                      padding: "0.5rem 0.6rem",
                    }}
                  >
                    <div style={{ fontWeight: 600, fontFamily: "JetBrains Mono, monospace" }}>
                      {m.model_id}
                    </div>
                    <div
                      className="mono"
                      style={{
                        color: "var(--muted)",
                        fontSize: "0.78rem",
                        wordBreak: "break-all",
                        marginTop: "0.1rem",
                      }}
                    >
                      {m.path || "Path not available (legacy endpoint)"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div>
            <label>Mode</label>
            <select
              value={mode}
              onChange={(e) => {
                const nextMode = e.target.value as PipelineMode;
                setMode(nextMode);
                if (nextMode === "normalize" || nextMode === "classify") {
                  setPipeCompareMethods(false);
                }
              }}
            >
              <option value="full">full (classify → NER → normalize)</option>
              <option value="classify">classify only</option>
              <option value="ner">NER only</option>
              <option value="normalize">normalize only</option>
            </select>
          </div>
          <div>
            <label>Processor (inference)</label>
            <select
              value={processor}
              onChange={(e) => setProcessor(e.target.value as Processor)}
            >
              <option value="auto">auto</option>
              <option value="cuda">cuda</option>
              <option value="mps">mps</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div>
            <label>Relevance threshold (classification)</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={relevanceThreshold}
              onChange={(e) => setRelevanceThreshold(parseFloat(e.target.value))}
              disabled={mode === "ner" || mode === "normalize"}
            />
            <div className="tag" style={{ marginTop: "0.35rem" }}>
              {relevanceThreshold.toFixed(2)}
            </div>
          </div>
          <div>
            <label>NER model (Hugging Face id)</label>
            <select
              value={nerMethod}
              onChange={(e) => {
                const next = e.target.value as NerMethod;
                setNerMethod(next);
                if (next === "transformers") {
                  void checkNerModelCompatibility(nerModel, true);
                } else {
                  setNerModelCheck((prev) => ({
                    ...prev,
                    compatible: null,
                    detectedTasks: [],
                    message: "Bent mode ignores HF model id; optional dependency required.",
                  }));
                }
              }}
              style={{ marginBottom: "0.5rem" }}
            >
              <option value="transformers">HF token-classification</option>
              <option value="bent">Bent (.ann parser)</option>
            </select>
            <input
              value={nerModel}
              onChange={(e) => {
                const value = e.target.value;
                setNerModel(value);
                if (nerMethod === "transformers") {
                  void checkNerModelCompatibility(value, true);
                }
              }}
              disabled={nerMethod === "bent"}
              style={{ width: "100%" }}
            />
            <div style={{ marginTop: "0.35rem" }}>{renderModelCheckTag("NER model", nerModelCheck)}</div>
            {renderModelCheckMessage(nerModelCheck)}
            {nerMethod === "bent" && (
              <p style={{ color: "var(--muted)", fontSize: "0.78rem", marginTop: "0.4rem" }}>
                Bent mode uses an optional dependency (`bent`) and ignores the model id.
                Install with {`pip install bent==0.0.80`} (Python 3.10.x, &lt;=3.10.13) and enable via:
                <br />
                - `./scripts/setup_bent_runtime.sh` + run backend from that venv, or
                <br />
                - start Bent service (`./scripts/run_bent_service.sh` or compose service) and set `BENT_SERVICE_URL`
              </p>
            )}
          </div>
          <div>
            <label>Compare methods</label>
            <select
              value={pipeCompareMethods ? "on" : "off"}
              onChange={(e) => setPipeCompareMethods(e.target.value === "on")}
              disabled={mode === "normalize" || mode === "classify"}
            >
              <option value="off">off</option>
              <option value="on">on (run both HF + Bent)</option>
            </select>
          </div>
          <div>
            <label>Batch size</label>
            <input
              type="number"
              min={1}
              max={64}
              value={pipeBatchSize}
              onChange={(e) => setPipeBatchSize(parseInt(e.target.value, 10))}
            />
          </div>
          <div>
            <label>Wikipedia fallback</label>
            <select
              value={pipeUseWikipedia ? "on" : "off"}
              onChange={(e) => setPipeUseWikipedia(e.target.value === "on")}
            >
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </div>
        </div>

        <div style={{ marginTop: "1rem" }}>
          <label>Articles JSON</label>
          <textarea
            value={pipeJson}
            onChange={(e) => setPipeJson(e.target.value)}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          />
        </div>

        {mode === "normalize" && (
          <div style={{ marginTop: "1rem" }}>
            <label>Mentions JSON (for normalize-only)</label>
            <textarea
              value={normJson}
              onChange={(e) => setNormJson(e.target.value)}
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            />
          </div>
        )}

        <div className="toolbar" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={applyPipelinePreset}
          >
            Apply MPS safe preset
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!isBackendConnected || busy || !modelId}
            onClick={runPipe}
          >
            Run pipeline
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy}
            onClick={() => jumpToStep("integrations")}
          >
            Back
          </button>
        </div>

        {lastRunInfo && lastRunInfo.files.length > 0 && (
          <div style={{ marginTop: "1rem" }}>
            <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
              Saved under <span className="mono">{lastRunInfo.path}</span>:{" "}
              {lastRunInfo.files.join(", ")}
            </p>
            <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
              Export last run (CSV needs only that step; Excel/PKL per table).
            </p>
            <div
              className="toolbar"
              style={{ marginTop: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}
            >
              {lastRunInfo.files.includes("classification.csv") && (
                <>
                  <span className="tag">classification</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "csv",
                          "classification.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "xlsx",
                          "classification.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "classification",
                          "pkl",
                          "classification.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              {lastRunInfo.files.includes("mentions.csv") && (
                <>
                  <span className="tag">mentions</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "csv",
                          "mentions.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "xlsx",
                          "mentions.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "mentions",
                          "pkl",
                          "mentions.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              {lastRunInfo.files.includes("normalized.csv") && (
                <>
                  <span className="tag">normalized</span>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "csv",
                          "normalized.csv"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    CSV
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "xlsx",
                          "normalized.xlsx"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    Excel
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy}
                    onClick={() =>
                      api
                        .downloadExport(
                          projectId,
                          "normalized",
                          "pkl",
                          "normalized.pkl"
                        )
                        .catch((err) => setErr((err as Error).message))
                    }
                  >
                    PKL
                  </button>
                </>
              )}
              <span className="tag">bundle</span>
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() =>
                  api
                    .downloadExport(
                      projectId,
                      "bundle",
                      "pkl",
                      "l2g_last_run_bundle.pkl"
                    )
                    .catch((err) => setErr((err as Error).message))
                }
              >
                All tables PKL
              </button>
              <button
                type="button"
                className="btn"
                disabled={busy}
                onClick={() =>
                  api
                    .downloadExport(
                      projectId,
                      "bundle",
                      "xlsx",
                      "l2g_last_run_bundle.xlsx"
                    )
                    .catch((err) => setErr((err as Error).message))
                }
              >
                All tables Excel
              </button>
            </div>
          </div>
        )}

        {pipeResult && (
          <div style={{ marginTop: "1rem" }}>
            {methodComparisonRows.length > 1 && (
              <div style={{ marginBottom: "1rem" }}>
                <h4>Method comparison (mentions)</h4>
                <div style={{ display: "grid", gap: "0.7rem" }}>
                  {methodComparisonRows.map((item) => (
                    <div key={item.method} style={{ border: "1px solid var(--border)", borderRadius: 8, padding: "0.6rem" }}>
                      <div className="tag" style={{ marginBottom: "0.4rem" }}>
                        {item.method === "transformers" ? "HF transformers" : "Bent (.ann)"} -{" "}
                        {item.rows.length} mentions
                      </div>
                      <pre
                        className="mono"
                        style={{ maxHeight: "160px", overflow: "auto", margin: 0, padding: "0.5rem", fontSize: "0.76rem" }}
                      >
                        {JSON.stringify(item.rows.slice(0, 30), null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <h4>Charts</h4>
            <ChartFromPayload pipeResult={pipeResult} />
            {filteredClassificationRows.length > 0 && (
              <div style={{ marginTop: "1rem" }}>
                <h4>
                  {`Classification rows filtered by probability (>= ${relevanceThreshold.toFixed(2)})`}
                </h4>
                <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
                  {filteredClassificationRows.length} / {classificationRows.length}{" "}
                  rows kept.
                </p>
                <pre
                  className="mono"
                  style={{
                    padding: "0.75rem",
                    background: "var(--bg)",
                    borderRadius: 8,
                    overflow: "auto",
                    maxHeight: "240px",
                    fontSize: "0.8rem",
                  }}
                >
                  {JSON.stringify(filteredClassificationRows, null, 2)}
                </pre>
              </div>
            )}
            <h4>Raw JSON</h4>
            <pre
              className="mono"
              style={{
                padding: "0.75rem",
                background: "var(--bg)",
                borderRadius: 8,
                overflow: "auto",
                maxHeight: "320px",
                fontSize: "0.8rem",
              }}
            >
              {JSON.stringify(pipeResult, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function ChartFromPayload({
  pipeResult,
}: {
  pipeResult: Record<string, unknown>;
}) {
  const charts = pipeResult.charts as Record<string, Record<string, number>> | undefined;
  if (!charts || typeof charts !== "object") {
    return <p style={{ color: "var(--muted)" }}>No chart data.</p>;
  }
  const entries = Object.entries(charts);
  if (entries.length === 0) {
    return null;
  }
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {entries.map(([title, obj]) => {
        const pairs = Object.entries(obj);
        const max = Math.max(1, ...pairs.map(([, v]) => v));
        return (
          <div key={title}>
            <div className="tag" style={{ marginBottom: "0.5rem" }}>
              {title}
            </div>
            <div className="bar-chart">
              {pairs.map(([k, v]) => (
                <div key={k} className="bar" style={{ height: `${(v / max) * 100}%` }}>
                  <span>{k}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
