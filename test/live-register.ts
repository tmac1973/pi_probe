/**
 * End-to-end registration test: load the real extension factory, capture the
 * registerProvider call the probe tool makes, and print the exact Pi model
 * configs that would be registered — against the live servers.
 */
const { default: factory } = await import("../src/index.ts");

const captured: Array<{ id: string; config: Record<string, unknown> }> = [];
let probeTool: { execute: Function } | undefined;

factory({
	registerCommand: () => {},
	registerTool: (tool: { name: string; execute: Function }) => {
		if (tool.name === "probe_inference_server") probeTool = tool;
	},
	on: () => () => {},
	registerProvider: (id: string, config: Record<string, unknown>) => {
		captured.push({ id, config });
	},
	unregisterProvider: () => {},
	registerVirtualModel: () => {},
	setModel: async () => true,
} as never);

if (!probeTool) throw new Error("probe tool not registered");

const fakeCtx = { modelRegistry: { find: () => ({}) } };

for (const url of process.argv.slice(2)) {
	captured.length = 0;
	const res = await probeTool.execute("tc", { url }, undefined, undefined, fakeCtx);
	const text = (res.content as Array<{ text: string }>)[0].text;
	console.log(`\n=== ${url} ===`);
	console.log(text);
	const prov = captured[0];
	if (!prov) {
		console.log("  (no provider registered)");
		continue;
	}
	console.log(`\n  provider id: ${prov.id}`);
	console.log(`  baseUrl: ${prov.config.baseUrl}  api: ${prov.config.api}`);
	for (const m of prov.config.models as Array<Record<string, unknown>>) {
		console.log(`  model ${m.id}:`);
		console.log(
			`    contextWindow=${m.contextWindow}  maxTokens=${m.maxTokens}  input=${JSON.stringify(m.input)}  reasoning=${m.reasoning}`,
		);
		if (m.samplingParams) console.log(`    samplingParams=${JSON.stringify(m.samplingParams)}`);
		if (m.thinkingLevelMap) console.log(`    thinkingLevelMap=${JSON.stringify(m.thinkingLevelMap)}`);
		if (m.compat) console.log(`    compat=${JSON.stringify(m.compat)}`);
	}
}
process.exit(0);
