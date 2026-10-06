/**
 * Extension factory smoke test: call the factory with a mock ExtensionAPI and
 * assert it registers the /probe command and probe_inference_server tool
 * without throwing.
 */
// Load the extension factory directly (Node >=23 strips TS types natively).
const { default: factory } = await import("../src/index.ts");

const registered: { commands: string[]; tools: string[] } = { commands: [], tools: [] };
const mockPi = {
	registerCommand: (name: string, opts: unknown) => {
		registered.commands.push(name);
		if (typeof (opts as { handler?: unknown }).handler !== "function") throw new Error("command missing handler");
	},
	registerTool: (tool: { name: string; execute?: unknown; parameters?: unknown }) => {
		registered.tools.push(tool.name);
		if (typeof tool.execute !== "function") throw new Error("tool missing execute");
		if (!tool.parameters) throw new Error("tool missing parameters");
	},
	on: () => () => {},
	registerProvider: () => {},
	unregisterProvider: () => {},
	registerVirtualModel: () => {},
};

factory(mockPi as never);

console.assert(registered.commands.includes("probe"), "registers /probe");
console.assert(registered.tools.includes("probe_inference_server"), "registers probe tool");
console.log("registered:", JSON.stringify(registered));
console.log("FACTORY SMOKE TEST PASSED");
process.exit(0);
