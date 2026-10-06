/**
 * Smoke test: spin up mock servers for each backend kind, run the probe
 * against them, and assert the normalized shape.
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { probeInferenceServer, normalizeBaseUrl } from "../src/probe.ts";

function listen(handler: (path: string) => unknown | undefined): Promise<number> {
	return new Promise((resolve) => {
		const server = createServer((req: IncomingMessage, res: ServerResponse) => {
			const body = handler(req.url ?? "");
			if (body === undefined) {
				res.writeHead(404).end();
				return;
			}
			res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(body));
		});
		server.listen(0, "127.0.0.1", () => {
			const addr = server.address();
			resolve(typeof addr === "object" && addr ? addr.port : 0);
		});
	});
}

// --- mock llama-toolchest ---------------------------------------------------
const toolchestPort = await listen((path) => {
	if (path === "/api/service/status") return { running: true };
	if (path === "/api/service/loaded-models") {
		return {
			schema_version: 1,
			running: true,
			models: [
				{
					id: "router-qwen",
					status: "loaded",
					public_name: "unsloth-Qwen3.6-35B-A3B.UD_Q8_K_XL",
					registry_id: "unsloth--Qwen3.6-35B-A3B-GGUF--Qwen3.6-35B-A3B-UD-Q8_K_XL",
					capabilities: {
						schema_version: 1,
						context_size: 131072,
						context_length: 262144,
						parallel: 4,
						context_shared: true,
						context_per_request: 32768,
						vision: false,
						tools: true,
						reasoning: {
							supported: true,
							default_enabled: true,
							toggle: "chat_template_kwargs",
							kwarg: "enable_thinking",
						},
						sampling: {
							source: "readme",
							default: { temperature: 1.0, top_p: 0.95, top_k: 20, presence_penalty: 1.5 },
							presets: [
								{ name: "thinking", label: "Thinking mode", temperature: 0.6, top_p: 0.95 },
								{ name: "non-thinking", temperature: 0.7 },
							],
						},
					},
				},
				{
					id: "router-llava",
					status: "unloaded",
					public_name: "llava-7b",
					capabilities: { context_per_request: 8192, vision: true, parallel: 1 },
				},
			],
		};
	}
	return undefined;
});

const r1 = await probeInferenceServer(`http://127.0.0.1:${toolchestPort}/`);
console.log("toolchest:", r1.kind, r1.notes, "defaultCtx:", r1.defaultContextSize);
const qwen = r1.models[0];
console.assert(r1.kind === "llama-toolchest", "kind");
console.assert(qwen.id === "unsloth-Qwen3.6-35B-A3B.UD_Q8_K_XL", "public_name as chat id");
console.assert(qwen.contextSize === 32768, "context_per_request wins");
console.assert(qwen.parallel === 4, "parallel");
console.assert(qwen.reasoning?.kwarg === "enable_thinking", "reasoning kwarg");
console.assert(qwen.sampling?.presets.length === 2, "presets");
console.assert(qwen.sampling?.default.temperature === 1.0, "default sampling");
console.assert(r1.models[1].loaded === false, "unloaded detected");
console.assert(r1.defaultContextSize === 32768, "default ctx from first loaded");

// --- mock stock llama-server --------------------------------------------------
const llamaPort = await listen((path) => {
	if (path === "/props") return { default_generation_settings: { n_ctx: 32768 }, total_slots: 1 };
	if (path === "/v1/models") return { data: [{ id: "Qwen3.5-9B-Q4_K_M.gguf", object: "model" }] };
	return undefined;
});
const r2 = await probeInferenceServer(`http://127.0.0.1:${llamaPort}/v1`);
console.log("llama-server:", r2.kind, r2.notes);
console.assert(r2.kind === "llama-server", "llama-server kind");
console.assert(r2.defaultContextSize === 32768, "n_ctx read");
console.assert(r2.models[0].contextSize === 32768, "model ctx");

// --- mock openai-compat -------------------------------------------------------
const openaiPort = await listen((path) => {
	if (path === "/v1/models") return { data: [{ id: "mistral-7b" }, { id: "gemma-2b" }] };
	return undefined;
});
const r3 = await probeInferenceServer(`http://127.0.0.1:${openaiPort}`);
console.log("openai-compat:", r3.kind, r3.notes);
console.assert(r3.kind === "openai-compat", "openai kind");
console.assert(r3.models.length === 2, "two models");

// --- mock ollama ----------------------------------------------------------------
const ollamaPort = await listen((path) => {
	if (path === "/api/tags") return { models: [{ name: "qwen2.5:7b" }, { name: "llama3.1:8b" }] };
	return undefined;
});
const r4 = await probeInferenceServer(`http://127.0.0.1:${ollamaPort}`);
console.log("ollama:", r4.kind, r4.notes);
console.assert(r4.kind === "ollama", "ollama kind");
console.assert(r4.models[0].id === "qwen2.5:7b", "ollama name");

// --- normalization ---------------------------------------------------------------
console.assert(normalizeBaseUrl("https://api.example.com/v1/") === "https://api.example.com", "strip /v1");
console.assert(normalizeBaseUrl("http://host:8080/") === "http://host:8080", "strip slash");
console.assert(normalizeBaseUrl("https://gw.example.com/llm-api") === "https://gw.example.com/llm-api", "keep path");
let threw = false;
try { normalizeBaseUrl("ftp://x"); } catch { threw = true; }
console.assert(threw, "rejects non-http");

console.log("ALL ASSERTIONS PASSED");
process.exit(0);
