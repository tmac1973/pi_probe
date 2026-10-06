/** Live probe against compute2:3000 (llama-toolchest) and compute:3000 (vllm). */
import { probeInferenceServer } from "../src/probe.ts";

for (const url of process.argv.slice(2)) {
	try {
		const r = await probeInferenceServer(url);
		console.log(`\n=== ${url} ===`);
		console.log(`kind: ${r.kind}`);
		console.log(`notes: ${r.notes}`);
		console.log(`defaultContextSize: ${r.defaultContextSize}`);
		for (const m of r.models) {
			console.log(
				`  - ${m.id}  [${m.displayName}]  ctx=${m.contextSize ?? "?"}  vision=${m.visionSupported ?? "?"}  loaded=${m.loaded ?? "?"}  parallel=${m.parallel ?? "?"}`,
			);
			if (m.reasoning) console.log(`      reasoning: ${JSON.stringify(m.reasoning)}`);
			if (m.sampling) console.log(`      sampling: ${JSON.stringify(m.sampling)}`);
		}
	} catch (e) {
		console.error(`\n=== ${url} === FAILED:`, e instanceof Error ? e.message : e);
	}
}
process.exit(0);
