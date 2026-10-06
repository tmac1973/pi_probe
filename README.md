# pi-inference-probe

Haruspex-style inference server discovery for the [Pi coding agent](https://pi.dev).

Point Pi at a running inference server — **llama-toolchest**, stock **llama-server**, any
**OpenAI-compatible** endpoint (LM Studio, vLLM, SGLang, Lemonade, llamafile), or **Ollama** —
and its models become selectable in Pi. No hand-editing `models.json`, no `/login` gymnastics.

The probe walks the same detection chain as Haruspex, most-informative first:

1. **llama-toolchest** — `GET /api/service/status` confirms the management layer, then a single
   `GET /api/service/loaded-models` yields rich per-model metadata: per-request context size,
   vision, parallel slots, loaded/unloaded state, reasoning toggle, recommended sampling.
2. **stock llama-server** — `GET /props` for `n_ctx`, then `/v1/models` for the list.
3. **generic OpenAI-compat** — `GET /v1/models`.
4. **Ollama native** — `GET /api/tags` (chat still routes through `/v1/chat/completions`).

## Install

Pi loads TypeScript extensions from `~/.pi/agent/extensions/` with no compile step (jiti):

```sh
# from this directory
pi install ./
```

or, without registering it in settings:

```sh
cp -r . ~/.pi/agent/extensions/pi-inference-probe
```

or for a one-off session:

```sh
pi --extension ./src/index.ts
```

Then run `/reload` inside Pi (or restart it). You should see the `probe_inference_server` tool
available and `/probe` in the command list.

## Usage

### `/probe` command

```
/probe http://192.168.1.10:8080            # probe + register + set as session model
/probe http://192.168.1.10:8080 my-key     # with a bearer key
/probe 192.168.1.10-8080                   # reconnect a saved server (by name or URL)
/probe                                      # re-probe last URL (or prompt for one)
/probe list                                 # show saved servers
/probe remove 192.168.1.10-8080             # unregister a saved server
/probe clear                                # hide the probe widget above the editor
/probe help                                 # show usage
```

Saved servers persist across restarts, but provider registration is per-session — after
restarting Pi, reconnect with `/probe <name>` (the name is host-port, e.g. `compute-3000`).

The probe result shows as a **compact one-line widget** above the editor, e.g.
`⚡ inference-compute-3000 · ThinkingCap-3.8-27B-PARO5 · session: …`. It stays until you
dismiss it with `/probe clear` (or it's replaced by the next probe). Run `/probe list` for the
full saved-server detail.

A probe does three things:

1. **Registers a live Pi provider** named `inference-<host>-<port>` whose models come straight
   from the server. Pick one with `/model` (search the provider name).
2. **Sets the first loaded model as the session model** so you can start typing immediately.
3. **Saves the server** to `~/.pi/agent/inference-probe.json` so `/probe` with no args re-probes
   it, and `/probe list` shows what you have.

### Agent tool

The agent itself can call `probe_inference_server` with `{ url, api_key? }` — so you can just
tell it "connect to the toolchest on 10.0.0.5" and it handles the rest.

## llama-toolchest specifics

Toolchest's `capabilities` block (schema v1) is mapped onto Pi model metadata:

| toolchest capability | where it lands in Pi |
|---|---|
| `context_per_request` (falls back to `context_size`) | model `contextWindow` — Pi compacts against this |
| `vision` | `input: ["text", "image"]` when true |
| `reasoning.toggle == "chat_template_kwargs"` + `kwarg` | `compat: { thinkingFormat: "chat-template", chatTemplateKwargs: { <kwarg>: { $var: "thinking.enabled" } } }` — Pi sends `enable_thinking: true/false` with the `/thinking` level |
| `reasoning.toggle == "reasoning_effort"` | `compat: { supportsReasoningEffort: true }` — Pi sends `reasoning_effort` |
| `reasoning.effort_levels` | `thinkingLevelMap` — levels the template doesn't accept are marked unsupported |
| `sampling.default` | model `samplingParams` (temperature / top_p / top_k / presence_penalty) |
| `loaded` / `status` | unloaded models are still registered but skipped when auto-selecting the session model |
| `public_name` | used as both the chat-completions model ID and the display name |

Notes and limits:

- **Sampling presets** (thinking / non-thinking bundles) are surfaced in the probe output but not
  wired to Pi's per-thinking-level `samplingParamsByThinkingLevel` — that would need a
  preset→thinking-level mapping policy. The `default` preset is applied.
- **`max_tokens`** is derived per model: reasoning models get `min(32768, context/4)` (they share
  the output budget between thinking and answer, so a tight cap truncates the reply), non-reasoning
  models get `min(8192, context/4)`. Override globally with the `PI_PROBE_MAX_TOKENS` env var.
- **Context size for non-toolchest backends**: llama-server reads `n_ctx` from `/props`;
  OpenAI-compat and Ollama fall back to 32768. If your server runs a bigger context, re-probe
  after the server reports it (or edit the provider's models via `models.json` overrides).
- The provider is registered **per Pi session** (Pi provider registrations are not persisted);
  the saved-server list is what survives, and re-probing is one command.

## Development

```sh
npm install
npm run check          # tsc --noEmit
node test/probe.test.ts    # probe logic against mock servers (all 4 backend kinds)
node test/factory.test.ts  # extension factory registers command + tool
```

Requires Node ≥ 23 for the test scripts (native TS type-stripping); Pi itself loads the
extension via jiti, so no build step is needed at runtime.

## Layout

```
src/probe.ts     detection chain + normalization (port of haruspex src-tauri/src/inference.rs)
src/index.ts     Pi extension: /probe command, probe_inference_server tool, provider registration
test/            mock-server smoke tests
```

## License

[MIT](./LICENSE)
