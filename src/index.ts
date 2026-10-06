/**
 * pi-inference-probe — Haruspex-style inference server discovery for Pi.
 *
 * Probes a local/LAN inference server (llama-toolchest, stock llama-server,
 * any OpenAI-compat endpoint, Ollama) and registers the discovered models
 * as a live Pi provider — no hand-editing models.json.
 *
 * Commands:
 *   /probe <url> [api-key]   probe + register + set as session model
 *   /probe <name>            reconnect to a previously saved server
 *   /probe                   re-probe the last URL (or prompt for one)
 *   /probe list              show saved servers
 *   /probe remove <name>     unregister a saved server
 *   /probe clear             hide the probe widget
 *   /probe help              show usage
 *
 * Tool:
 *   probe_inference_server   the agent can discover/register servers itself
 *
 * llama-toolchest gets first-class treatment: its /api/service/loaded-models
 * capabilities block (per-request context, vision, parallel slots, reasoning
 * toggle, recommended sampling) is mapped onto Pi model metadata, including
 * chat_template_kwargs wiring for Qwen-style thinking toggles.
 */

import { Type } from "typebox";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import {
	probeInferenceServer,
	normalizeBaseUrl,
	type NormalizedModel,
	type ProbeResult,
	type ReasoningCaps,
	type SamplingCaps,
} from "./probe.ts";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROVIDER_PREFIX = "inference-";
const PROVIDER_ID_MAX = 32; // pi-ai provider IDs are [a-z0-9-]{1,32}
const STATE_FILE = "inference-probe.json";
const MAX_OUTPUT_TOKENS = 8192;
// Reasoning models share the output budget between thinking and answer, so a
// tight cap truncates the reply once the thinking phase gets long. Give them
// more room while staying well under the context window.
const MAX_OUTPUT_TOKENS_REASONING = 32768;

// ---------------------------------------------------------------------------
// Saved server state (persisted in the agent dir, outside the session)
// ---------------------------------------------------------------------------

interface SavedServer {
	name: string;
	baseUrl: string;
	apiKey?: string;
	kind: string;
	notes: string;
	probedAt: number;
}

function statePath(): string {
	return join(process.env.PI_AGENT_DIR ?? `${process.env.HOME}/.pi/agent`, STATE_FILE);
}

function loadState(): SavedServer[] {
	try {
		const parsed = JSON.parse(readFileSync(statePath(), "utf8")) as { servers?: SavedServer[] };
		return Array.isArray(parsed.servers) ? parsed.servers : [];
	} catch {
		return [];
	}
}

function saveState(servers: SavedServer[]): void {
	try {
		mkdirSync(dirname(statePath()), { recursive: true });
		writeFileSync(statePath(), JSON.stringify({ servers }, null, 2));
	} catch {
		// Non-fatal: worst case the user re-probes.
	}
}

function nameFor(baseUrl: string): string {
	const u = new URL(baseUrl);
	return `${u.hostname.replace(/[^a-z0-9.-]/gi, "-")}-${u.port || (u.protocol === "https:" ? "443" : "80")}`;
}

function providerIdFor(baseUrl: string): string {
	const name = nameFor(baseUrl);
	const id = name.length <= PROVIDER_ID_MAX ? name : `${name.slice(0, 24)}-${crypto.randomUUID().slice(0, 8)}`;
	return `${PROVIDER_PREFIX}${id}`;
}

// ---------------------------------------------------------------------------
// Probe result -> Pi model metadata
// ---------------------------------------------------------------------------

function defaultMaxTokens(m: NormalizedModel): number {
	const cap = m.reasoning?.supported ? MAX_OUTPUT_TOKENS_REASONING : MAX_OUTPUT_TOKENS;
	const env = Number(process.env.PI_PROBE_MAX_TOKENS);
	if (Number.isFinite(env) && env > 0) return Math.min(env, m.contextSize ?? Number.MAX_SAFE_INTEGER);
	const ctx = m.contextSize;
	if (ctx === undefined) return cap;
	return Math.min(cap, Math.max(2048, Math.floor(ctx / 4)));
}

function samplingFor(m: NormalizedModel): Record<string, unknown> | undefined {
	const s: SamplingCaps | undefined = m.sampling;
	if (!s) return undefined;
	const out: Record<string, unknown> = { ...s.default };
	for (const key of ["temperature", "top_p", "top_k", "presence_penalty"] as const) {
		if (out[key] === undefined) delete out[key];
	}
	return Object.keys(out).length > 0 ? out : undefined;
}

function thinkingLevelMapFor(m: NormalizedModel): Record<string, string | null> | undefined {
	const r: ReasoningCaps | undefined = m.reasoning;
	if (!r?.supported) return undefined;
	const map: Record<string, string | null> = {};
	if (r.effort_levels) {
		for (const level of ["off", "minimal", "low", "medium", "high", "xhigh", "max"]) {
			map[level] = r.effort_levels.includes(level) ? level : null;
		}
	}
	return Object.keys(map).length > 0 ? map : undefined;
}

function compatFor(m: NormalizedModel): Record<string, unknown> | undefined {
	const r: ReasoningCaps | undefined = m.reasoning;
	if (r?.supported && r.toggle === "chat_template_kwargs" && r.kwarg) {
		// Pi's openai-completions adapter sends chat_template_kwargs with
		// $var: "thinking.enabled" -> true when a thinking level is active,
		// false otherwise — exactly what Qwen-style templates expect.
		return { thinkingFormat: "chat-template", chatTemplateKwargs: { [r.kwarg]: { $var: "thinking.enabled" } } };
	}
	if (r?.supported && r.toggle === "reasoning_effort") {
		// OpenAI-style reasoning_effort field (gpt-oss and friends).
		return { supportsReasoningEffort: true };
	}
	return undefined;
}

function toModelConfig(m: NormalizedModel): ProviderModelConfig {
	const config: ProviderModelConfig = {
		id: m.id,
		name: m.displayName !== m.id ? `${m.displayName} (${m.id})` : m.id,
		input: m.visionSupported === true ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: m.contextSize ?? 32768,
		maxTokens: defaultMaxTokens(m),
		reasoning: m.reasoning?.supported === true,
	};
	const sampling = samplingFor(m);
	if (sampling) config.samplingParams = sampling;
	const tlm = thinkingLevelMapFor(m);
	if (tlm) config.thinkingLevelMap = tlm;
	const compat = compatFor(m);
	if (compat) config.compat = compat as never;
	return config;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

function registerServer(pi: ExtensionAPI, result: ProbeResult, apiKey?: string): string {
	const providerId = providerIdFor(result.baseUrl);
	// Re-registering the same ID replaces the earlier registration.
	pi.registerProvider(providerId, {
		name: `Inference: ${nameFor(result.baseUrl)}`,
		baseUrl: `${result.baseUrl}/v1`,
		apiKey: apiKey?.trim() ? apiKey.trim() : "local",
		api: "openai-completions",
		models: result.models.map(toModelConfig),
	});
	return providerId;
}

function saveServer(result: ProbeResult, apiKey?: string): void {
	const servers = loadState().filter((s) => s.baseUrl !== result.baseUrl);
	servers.push({
		name: nameFor(result.baseUrl),
		baseUrl: result.baseUrl,
		apiKey: apiKey?.trim() || undefined,
		kind: result.kind,
		notes: result.notes,
		probedAt: Date.now(),
	});
	saveState(servers);
}

function formatProbeResult(result: ProbeResult, providerId: string): string {
	const lines: string[] = [
		`Probed ${result.baseUrl} — ${result.notes}`,
		`Registered as provider \`${providerId}\` with ${result.models.length} model(s):`,
	];
	for (const m of result.models) {
		const bits = [
			m.loaded === false ? "unloaded" : undefined,
			m.contextSize ? `ctx=${m.contextSize}` : undefined,
			m.visionSupported ? "vision" : undefined,
			m.parallel && m.parallel > 1 ? `parallel=${m.parallel}` : undefined,
			m.reasoning?.supported ? `reasoning(${m.reasoning.toggle}${m.reasoning.kwarg ? `:${m.reasoning.kwarg}` : ""})` : undefined,
			m.sampling?.default.temperature !== undefined ? `temp=${m.sampling.default.temperature}` : undefined,
		].filter(Boolean);
		lines.push(`  - ${m.id}${bits.length ? `  [${bits.join(", ")}]` : ""}`);
	}
	lines.push("");
	lines.push("Pick one with /model (search the provider name), or it is already set as the session model if it was the first.");
	return lines.join("\n");
}

function formatProbeSummary(result: ProbeResult, providerId: string, modelId?: string): string {
	const loaded = result.models.filter((m) => m.loaded !== false).length;
	const models =
		result.models.length === 1 ? result.models[0].id : `${result.models.length} models (${loaded} loaded)`;
	return `⚡ ${providerId} · ${models}${modelId ? ` · session: ${modelId}` : ""} · /probe list for details`;
}

function formatProbeHelp(): string[] {
	return [
		"/probe — connect Pi to a running inference server",
		"",
		"  /probe <url> [api-key]   probe a server, register its models, set as session model",
		"  /probe <name>            reconnect to a saved server (by name or URL)",
		"  /probe                   re-probe the last server (or prompt for one)",
		"  /probe list              show all saved servers",
		"  /probe remove <name>     unregister + forget a saved server",
		"  /probe clear             hide the probe widget",
		"  /probe help              this help",
		"",
		"  e.g.  /probe http://compute:3000      /probe compute-3000",
		"",
		"Saved servers persist across restarts; provider registration is per-session,",
		"so reconnect after each restart with /probe <name>.",
	];
}

async function setFirstModel(pi: ExtensionAPI, ctx: ExtensionCommandContext | ExtensionToolContext, result: ProbeResult, providerId: string): Promise<string | undefined> {
	const usable = result.models.find((m) => m.loaded !== false) ?? result.models[0];
	if (!usable) return undefined;
	const model = ctx.modelRegistry.find(providerId, usable.id);
	if (!model) return undefined;
	const ok = await pi.setModel(model);
	return ok ? usable.id : undefined;
}

// ---------------------------------------------------------------------------
// Command + tool
// ---------------------------------------------------------------------------

async function runProbe(
	pi: ExtensionAPI,
	ctx: ExtensionCommandContext | ExtensionToolContext,
	urlArg: string,
	apiKey?: string,
): Promise<{ text: string; result?: ProbeResult; providerId?: string; modelId?: string }> {
	let result: ProbeResult;
	try {
		result = await probeInferenceServer(urlArg, apiKey);
	} catch (err) {
		return { text: `Probe failed: ${err instanceof Error ? err.message : String(err)}` };
	}
	const providerId = registerServer(pi, result, apiKey);
	saveServer(result, apiKey);
	const modelId = await setFirstModel(pi, ctx, result, providerId);
	return {
		text: formatProbeResult(result, providerId) + (modelId ? `\nSession model set to ${providerId}/${modelId}.` : ""),
		result,
		providerId,
		modelId,
	};
}

export default function (pi: ExtensionAPI) {
	pi.registerCommand("probe", {
		description:
			"Probe an inference server and register its models: /probe <url> [api-key], /probe <name> to reconnect a saved server, /probe list, /probe remove <name>, /probe clear, /probe help",
		async handler(args, ctx) {
			const trimmed = args.trim();
			if (!ctx.hasUI) {
				ctx.ui.notify("probe needs an interactive UI", "error");
				return;
			}
			if (!trimmed) {
				const last = loadState().at(-1);
				const url =
					(await ctx.ui.input("Inference server URL", last ? `last: ${last.baseUrl}` : "http://192.168.1.10:8080")) ?? "";
				if (!url.trim()) return;
				const key = (await ctx.ui.input("API key (blank for none)")) ?? "";
				const out = await runProbe(pi, ctx, url.trim(), key.trim() || undefined);
				if (out.result && out.providerId) {
					ctx.ui.setWidget("probe", [formatProbeSummary(out.result, out.providerId, out.modelId)]);
				} else {
					ctx.ui.notify(out.text, "error");
				}
				return;
			}
			if (trimmed === "list") {
				const servers = loadState();
				if (servers.length === 0) {
					ctx.ui.notify("No saved servers. Use /probe <url> first.", "info");
					return;
				}
				ctx.ui.setWidget(
					"probe",
					servers.map(
						(s) =>
							`${s.name}  ${s.baseUrl}  [${s.kind}]  ${new Date(s.probedAt).toLocaleString()}  — ${s.notes}`,
					),
				);
				return;
			}
			if (trimmed === "clear") {
				ctx.ui.setWidget("probe", undefined);
				ctx.ui.notify("Probe widget cleared", "info");
				return;
			}
			if (trimmed === "help" || trimmed === "?") {
				ctx.ui.setWidget("probe", formatProbeHelp());
				return;
			}
			if (trimmed.startsWith("remove ")) {
				const name = trimmed.slice(7).trim();
				const servers = loadState();
				const idx = servers.findIndex((s) => s.name === name || s.baseUrl === name);
				if (idx === -1) {
					ctx.ui.notify(`No saved server named "${name}"`, "warning");
					return;
				}
				const [removed] = servers.splice(idx, 1);
				saveState(servers);
				try {
					pi.unregisterProvider(providerIdFor(removed.baseUrl));
				} catch {
					// Not registered this session; nothing to do.
				}
				ctx.ui.notify(`Removed ${removed.name}`, "info");
				return;
			}
			// /probe <url|name> [api-key] — probe a new server or reconnect a saved one
			const parts = trimmed.split(/\s+/);
			const arg = parts[0];
			const key = parts[1];
			// Normalize the arg so "compute:3000", "http://compute:3000", and
			// "http://compute:3000/" all compare equal.
			const norm = (s: string) => s.replace(/^https?:\/\//, "").replace(/\/+$/, "").toLowerCase();
			const saved = loadState().find((s) => s.name === arg || norm(s.baseUrl) === norm(arg));
			const out = await runProbe(pi, ctx, saved ? saved.baseUrl : arg, key ?? saved?.apiKey);
			if (out.result && out.providerId) {
				ctx.ui.setWidget("probe", [formatProbeSummary(out.result, out.providerId, out.modelId)]);
			} else {
				ctx.ui.notify(out.text, "error");
			}
		},
	});

	pi.registerTool({
		name: "probe_inference_server",
		label: "Probe inference server",
		description:
			"Discover and register a local/LAN inference server (llama-toolchest, llama-server, OpenAI-compatible, Ollama). " +
			"Probes the URL, registers the discovered models as a Pi provider, saves the server, and sets the first usable model as the session model. " +
			"Use when the user wants to connect Pi to a running inference server.",
		parameters: Type.Object({
			url: Type.String({ description: "Server base URL, e.g. http://192.168.1.10:8080 (a trailing /v1 is stripped)" }),
			api_key: Type.Optional(Type.String({ description: "Bearer API key, if the server requires one" })),
		}),
		annotations: { readOnlyHint: false },
		async execute(_id, params, _signal, _onUpdate, ctx) {
			const out = await runProbe(pi, ctx, params.url, params.api_key);
			return {
				content: [{ type: "text", text: out.text }],
				details: undefined,
			};
		},
	});
}
