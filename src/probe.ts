/**
 * Inference server discovery + probing.
 *
 * Ported from Haruspex's src-tauri/src/inference.rs. Walks four detection
 * strategies, most-informative first:
 *
 *   1. llama-toolchest — GET /api/service/status confirms the management
 *      layer, then GET /api/service/loaded-models yields rich per-model
 *      metadata inline under a `capabilities` object: per-request context
 *      size, vision, parallel slots, reasoning toggle, recommended sampling.
 *   2. stock llama-server — GET /props exposes n_ctx; /v1/models for the list.
 *   3. generic OpenAI-compat — GET /v1/models (LM Studio, vLLM, Ollama, ...).
 *   4. Ollama native — GET /api/tags.
 *
 * Regardless of which path hits, the probe returns one normalized shape so
 * the rest of the extension only understands that.
 */

export type BackendKind = "llama-toolchest" | "llama-server" | "openai-compat" | "ollama";

export interface SamplingParams {
	temperature?: number;
	top_p?: number;
	top_k?: number;
	presence_penalty?: number;
}

export interface SamplingPreset extends SamplingParams {
	name: string;
	label?: string;
}

export interface SamplingCaps {
	default: SamplingParams;
	presets: SamplingPreset[];
	source?: string;
}

export interface ReasoningCaps {
	supported: boolean;
	default_enabled: boolean;
	/** "chat_template_kwargs" | "reasoning_effort" | "none" */
	toggle: string;
	/** kwarg key (e.g. "enable_thinking") when toggle is chat_template_kwargs */
	kwarg?: string;
	/** Effort levels the template accepts, when the server enumerates them. */
	effort_levels?: string[];
	/** What the model does when no effort is sent. */
	default_effort?: string;
}

export interface NormalizedModel {
	/** The ID /v1/chat/completions accepts (toolchest: public_name). */
	id: string;
	displayName: string;
	/** Per-request context (toolchest: context_per_request). */
	contextSize?: number;
	visionSupported?: boolean;
	/** Only meaningful for llama-toolchest, which distinguishes loaded/unloaded. */
	loaded?: boolean;
	/** Parallel sequence slots the server runs this model with. */
	parallel?: number;
	reasoning?: ReasoningCaps;
	sampling?: SamplingCaps;
}

export interface ProbeResult {
	/** Normalized service root the probe succeeded against. */
	baseUrl: string;
	kind: BackendKind;
	models: NormalizedModel[];
	/** Backend-reported default context window, when detectable. */
	defaultContextSize?: number;
	/** Short human-readable note, e.g. "llama-toolchest (2 loaded of 3 enabled)". */
	notes: string;
}

const PROBE_TIMEOUT_MS = 6000;

/**
 * Normalize a user-entered base URL to a "service root" that downstream code
 * can append paths onto. Strips trailing slashes and a trailing /v1 segment.
 */
export function normalizeBaseUrl(input: string): string {
	const trimmed = input.trim();
	if (!trimmed) throw new Error("URL is empty");
	let url: URL;
	try {
		url = new URL(trimmed);
	} catch {
		throw new Error(`Invalid URL: ${trimmed}`);
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error(`Unsupported URL scheme: ${url.protocol}`);
	}
	if (!url.hostname) throw new Error("URL is missing a host");
	let root = url.toString().replace(/\/+$/, "");
	// Users often paste the OpenAI-compat base URL; detection needs the root.
	if (root.endsWith("/v1")) root = root.slice(0, -3).replace(/\/+$/, "");
	return root;
}

function authHeaders(apiKey?: string): Record<string, string> {
	const key = apiKey?.trim();
	return key ? { Authorization: `Bearer ${key}` } : {};
}

/** GET + JSON parse; undefined on any failure so endpoint errors degrade gracefully. */
async function fetchJson(url: string, apiKey?: string): Promise<unknown> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
	try {
		const resp = await fetch(url, { headers: authHeaders(apiKey), signal: controller.signal });
		if (!resp.ok) return undefined;
		return (await resp.json()) as unknown;
	} catch {
		return undefined;
	} finally {
		clearTimeout(timer);
	}
}

function asArray(v: unknown): unknown[] {
	return Array.isArray(v) ? v : [];
}
function asString(v: unknown): string | undefined {
	return typeof v === "string" && v.length > 0 ? v : undefined;
}
function asBool(v: unknown): boolean | undefined {
	return typeof v === "boolean" ? v : undefined;
}
function asNumber(v: unknown): number | undefined {
	return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function parseSamplingParams(v: unknown): SamplingParams {
	const o = (v ?? {}) as Record<string, unknown>;
	const out: SamplingParams = {};
	if (asNumber(o.temperature) !== undefined) out.temperature = o.temperature as number;
	if (asNumber(o.top_p) !== undefined) out.top_p = o.top_p as number;
	if (asNumber(o.top_k) !== undefined) out.top_k = o.top_k as number;
	if (asNumber(o.presence_penalty) !== undefined) out.presence_penalty = o.presence_penalty as number;
	return out;
}

function parseSamplingCaps(v: unknown): SamplingCaps | undefined {
	if (typeof v !== "object" || v === null) return undefined;
	const o = v as Record<string, unknown>;
	const presets: SamplingPreset[] = [];
	for (const p of asArray(o.presets)) {
		const po = p as Record<string, unknown>;
		const name = asString(po.name);
		if (!name) continue;
		presets.push({ ...parseSamplingParams(po), name, label: asString(po.label) });
	}
	return { default: parseSamplingParams(o.default), presets, source: asString(o.source) };
}

function parseReasoningCaps(v: unknown): ReasoningCaps | undefined {
	if (typeof v !== "object" || v === null) return undefined;
	const o = v as Record<string, unknown>;
	const toggle = asString(o.toggle);
	if (!toggle) return undefined;
	const rawLevels = asArray(o.effort_levels ?? o.levels).filter((x): x is string => typeof x === "string");
	return {
		supported: asBool(o.supported) ?? false,
		default_enabled: asBool(o.default_enabled) ?? false,
		toggle,
		kwarg: asString(o.kwarg),
		effort_levels: rawLevels.length > 0 ? rawLevels : undefined,
		default_effort: asString(o.default_effort ?? o.effort_default),
	};
}

function parseCapabilities(v: unknown): {
	contextSize?: number;
	vision?: boolean;
	parallel?: number;
	reasoning?: ReasoningCaps;
	sampling?: SamplingCaps;
} {
	if (typeof v !== "object" || v === null) return {};
	const o = v as Record<string, unknown>;
	// context_per_request already accounts for parallel-slot KV division — it's
	// the value the client must compact against. Fall back to the raw pool size.
	const contextSize = asNumber(o.context_per_request) ?? asNumber(o.context_size);
	return {
		contextSize,
		vision: asBool(o.vision),
		parallel: asNumber(o.parallel),
		reasoning: parseReasoningCaps(o.reasoning),
		sampling: parseSamplingCaps(o.sampling),
	};
}

function parseToolchestModelList(v: unknown): NormalizedModel[] {
	const root = (v ?? {}) as Record<string, unknown>;
	const arr = asArray(root.models).length > 0 ? asArray(root.models) : asArray(v);
	const out: NormalizedModel[] = [];
	for (const item of arr) {
		const o = item as Record<string, unknown>;
		const id = asString(o.id) ?? asString(o.model_id) ?? asString(o.name);
		if (!id) continue;
		// public_name is toolchest's short OpenAI-style ID — what /v1/models
		// advertises. Prefer it for display; it is also the chat ID when present.
		const name = asString(o.public_name) ?? asString(o.name) ?? asString(o.display_name) ?? id;
		const status = asString(o.status);
		const loaded = asBool(o.loaded) ?? (status ? ["loaded", "ready", "running"].includes(status) : undefined) ?? true;
		const caps = parseCapabilities(o.capabilities);
		out.push({
			id: asString(o.public_name) ?? id,
			displayName: name,
			contextSize: caps.contextSize,
			visionSupported: caps.vision,
			loaded,
			parallel: caps.parallel,
			reasoning: caps.reasoning,
			sampling: caps.sampling,
		});
	}
	return out;
}

function parseOpenaiModelList(v: unknown): string[] {
	const root = (v ?? {}) as Record<string, unknown>;
	const arr = asArray(root.data).length > 0 ? asArray(root.data) : asArray(root.models).length > 0 ? asArray(root.models) : asArray(v);
	const out: string[] = [];
	for (const item of arr) {
		const o = item as Record<string, unknown>;
		const id = asString(o.id) ?? asString(o.name);
		if (id) out.push(id);
	}
	return out;
}

function parseOllamaTags(v: unknown): NormalizedModel[] {
	const root = (v ?? {}) as Record<string, unknown>;
	return asArray(root.models)
		.map((item) => {
			const name = asString((item as Record<string, unknown>).name);
			return name ? { id: name, displayName: name } : undefined;
		})
		.filter((m): m is NormalizedModel => m !== undefined);
}

function plural(n: number): string {
	return n === 1 ? "" : "s";
}

async function tryLlamaToolchest(base: string, apiKey?: string): Promise<ProbeResult | undefined> {
	const status = await fetchJson(`${base}/api/service/status`, apiKey);
	if (status === undefined) return undefined;
	const loaded = await fetchJson(`${base}/api/service/loaded-models`, apiKey);
	if (loaded === undefined) return undefined;
	const models = parseToolchestModelList(loaded);
	let defaultContextSize: number | undefined;
	let firstContextSize: number | undefined;
	let loadedCount = 0;
	for (const m of models) {
		if (m.loaded) loadedCount++;
		if (firstContextSize === undefined) firstContextSize = m.contextSize;
		if (defaultContextSize === undefined && m.loaded) defaultContextSize = m.contextSize;
	}
	defaultContextSize ??= firstContextSize;
	return {
		baseUrl: base,
		kind: "llama-toolchest",
		models,
		defaultContextSize,
		notes: `llama-toolchest (${loadedCount} loaded of ${models.length} enabled)`,
	};
}

async function tryLlamaServer(base: string, apiKey?: string): Promise<ProbeResult | undefined> {
	const props = (await fetchJson(`${base}/props`, apiKey)) as Record<string, unknown> | undefined;
	if (props === undefined) return undefined;
	const dgs = (props.default_generation_settings ?? {}) as Record<string, unknown>;
	const nCtx = asNumber(dgs.n_ctx) ?? asNumber(props.n_ctx);
	const ids = parseOpenaiModelList(await fetchJson(`${base}/v1/models`, apiKey));
	if (ids.length === 0) return undefined;
	const models: NormalizedModel[] = ids.map((id) => ({
		id,
		displayName: id,
		contextSize: nCtx,
	}));
	return {
		baseUrl: base,
		kind: "llama-server",
		models,
		defaultContextSize: nCtx,
		notes: `llama-server (${models.length} model${plural(models.length)}${nCtx ? `, n_ctx=${nCtx}` : ""})`,
	};
}

async function tryOpenAiCompat(base: string, apiKey?: string): Promise<ProbeResult | undefined> {
	const ids = parseOpenaiModelList(await fetchJson(`${base}/v1/models`, apiKey));
	if (ids.length === 0) return undefined;
	return {
		baseUrl: base,
		kind: "openai-compat",
		models: ids.map((id) => ({ id, displayName: id })),
		notes: `OpenAI-compatible (${ids.length} model${plural(ids.length)})`,
	};
}

async function tryOllamaNative(base: string, apiKey?: string): Promise<ProbeResult | undefined> {
	const models = parseOllamaTags(await fetchJson(`${base}/api/tags`, apiKey));
	if (models.length === 0) return undefined;
	return {
		baseUrl: base,
		kind: "ollama",
		models,
		notes: `Ollama native (${models.length} model${plural(models.length)})`,
	};
}

/** Walk the detection chain; returns the richest backend that responds. */
export async function probeInferenceServer(baseUrl: string, apiKey?: string): Promise<ProbeResult> {
	const base = normalizeBaseUrl(baseUrl);
	for (const attempt of [tryLlamaToolchest, tryLlamaServer, tryOpenAiCompat, tryOllamaNative]) {
		const result = await attempt(base, apiKey);
		if (result) return result;
	}
	throw new Error(
		`Couldn't detect a supported inference server at ${base}. Tried: ` +
			"llama-toolchest (/api/service/status), llama-server (/props), OpenAI-compat (/v1/models), Ollama (/api/tags). " +
			"Check that the URL is correct and the server is reachable.",
	);
}
