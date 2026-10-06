/**
 * Integration test: session_start must auto-reconnect the most recently used
 * saved server — registering its provider WITHOUT setting the session model.
 * Runs against the real compute:3000 (read-only GETs); skips if unreachable.
 */
import { readFileSync } from "node:fs";
import ext from "../src/index.ts";

interface Calls {
	providers: string[];
	setModel: number;
	widget: unknown;
	handler: ((event: unknown, ctx: unknown) => void) | undefined;
}
const calls: Calls = { providers: [], setModel: 0, widget: undefined, handler: undefined };

const pi: never = {
	registerCommand: () => {},
	registerTool: () => {},
	on: (event: string, handler: (e: unknown, c: unknown) => void) => {
		if (event === "session_start") calls.handler = handler;
	},
	registerProvider: (id: string) => {
		calls.providers.push(id);
	},
	unregisterProvider: () => {},
	setModel: async () => {
		calls.setModel++;
		return true;
	},
};

const ctx = {
	mode: "tui",
	hasUI: true,
	// Fresh session with no default model: Pi holds the DEFAULT_MODEL
	// placeholder (provider "unknown") — auto-reconnect must adopt the
	// probed model, which is what makes typing work immediately.
	model: { provider: "unknown" },
	ui: {
		setWidget: (_key: string, content: unknown) => {
			calls.widget = content;
		},
		notify: () => {},
	},
	// A real registry would resolve the just-registered model; the mock
	// returns one so setFirstModel can complete and call pi.setModel.
	modelRegistry: { find: () => ({ id: "tcclaviger/ThinkingCap-3.8-27B-PARO5", provider: "inference-compute-3000" }) },
};

ext(pi);
if (!calls.handler) throw new Error("session_start handler not registered");
calls.handler({}, ctx);

// autoReconnect is detached — poll for the provider registration.
const deadline = Date.now() + 15_000;
while (calls.providers.length === 0 && Date.now() < deadline) {
	await new Promise((r) => setTimeout(r, 200));
}

const state = JSON.parse(
	readFileSync(process.env.PI_AGENT_DIR ? `${process.env.PI_AGENT_DIR}/inference-probe.json` : `${process.env.HOME}/.pi/agent/inference-probe.json`, "utf8"),
) as { servers: { name: string; lastUsed?: number; probedAt: number }[] };
const expected = state.servers
	.filter((s) => (s.lastUsed ?? s.probedAt) > 0)
	.sort((a, b) => (b.lastUsed ?? b.probedAt) - (a.lastUsed ?? a.probedAt))[0];

if (!expected) throw new Error("no saved servers in state file");
if (calls.providers.length !== 1) throw new Error(`expected 1 provider, got ${calls.providers.length} (server down?)`);
if (calls.providers[0] !== `inference-${expected.name}`)
	throw new Error(`expected provider inference-${expected.name}, got ${calls.providers[0]}`);
if (calls.setModel !== 1)
	throw new Error(`fresh session (provider "unknown") must adopt the probed model, setModel called ${calls.setModel}x`);
if (!Array.isArray(calls.widget) || calls.widget.length !== 1 || !String(calls.widget[0]).startsWith("⚡"))
	throw new Error(`expected compact one-line widget, got ${JSON.stringify(calls.widget)}`);

console.log("provider:", calls.providers[0]);
console.log("widget:  ", calls.widget[0]);
console.log("setModel calls:", calls.setModel);
console.log("AUTO-RECONNECT ASSERTIONS PASSED");
