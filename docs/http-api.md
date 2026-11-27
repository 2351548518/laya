# HTTP API

`POST /v1/systemone` uses the TypeSafe System One format for text, image decisions and
single-box localization. The default is the released Qwen3.5-4B decision checkpoint. Set
`LAYA_MULTIMODAL_MODEL` to choose another trained checkpoint, such as a Gemma model for
audio. Send audio and video file bytes to `POST /v1/multimodal`. [TypeSafe API](https://docs.typesafe.ai/api/system-one) ·
[Media examples](multimodal-clients.md)

The Python HTTP client provides synchronous single, batch and media predictions plus health
checks. Install it from `sdk/python` in the application environment to keep the inference stack
on the server. [Python SDK](https://github.com/2351548518/laya/tree/main/sdk/python)

```bash
python -m pip install -e '.[multimodal,serve]'
laya-serve            # http://0.0.0.0:8000
```

The same entry point runs embedded in any ASGI server: `laya.serve.create_app()` builds the FastAPI
app, optionally with a `Router` you inject (`create_app(router)`) instead of one built from the
environment.

## Configuration

Everything is environment variables, so one image serves a laptop dev run and a systemd unit.

| env var | meaning | default |
|---|---|---|
| `LAYA_HOST` | bind address | `0.0.0.0` |
| `LAYA_PORT` | bind port | `8000` |
| `LAYA_ROOT_PATH` | public URL prefix when served behind a reverse proxy | empty |
| `LAYA_DEVICE` | torch device for every checkpoint | auto |
| `LAYA_MULTIMODAL_MODEL` | trained native checkpoint directory or Hub ID | released Qwen3.5-4B |
| `LAYA_PRELOAD` | build the checkpoints at startup, not lazily | `1` |
| `LAYA_MODELS` | comma-separated checkpoint names to preload | registered checkpoints, or only `multimodal` when `LAYA_MULTIMODAL_MODEL` is set |
| `LAYA_THREADS` | cap torch intra-op threads on CPU; keep it <= physical cores -- oversubscribing logical cores is a large regression | torch default |
| `LAYA_IDLE_UNLOAD_SECONDS` | unload resident checkpoints after this many idle seconds; the next request loads its checkpoint again. Zero disables unloading | `0` |
| `LAYA_MAX_LOADED` | checkpoints kept resident at once; a cap below what routing chooses rebuilds one per switch, and `preload()` raises it to hold whatever it builds | `2` |
| `LAYA_MAX_TOKEN_BUDGET` | server-side ceiling on the per-request `max_len` and `head_max_len` overrides; a larger value is a `422`. Unparseable or non-positive input logs a warning and falls back | `8192` |
| `LAYA_DEFAULT_MODEL` | default registered checkpoint. `laya` and `default` resolve to `multimodal` | `multimodal` |
| `LAYA_EXTRA_MODELS` | JSON object mapping names to local native checkpoint directories or Hub repo IDs | empty |
| `LAYA_API_KEY` | if set, require `Authorization: Bearer <key>` on every route, and stop serving `/docs`, `/redoc` and `/openapi.json` (FastAPI's own routes, which per-route auth never covered) | none |
| `LAYA_LOG_LEVEL` | uvicorn log level | `info` |
| `LAYA_MAX_CONCURRENT` | requests admitted past auth at once; excess gets `503` | `16` |
| `LAYA_BODY_READ_TIMEOUT` | seconds a request may spend delivering its body before the server answers `408` and releases the admission slot it was holding. Unparseable, non-positive or infinite input falls back | `10` |
| `LAYA_MAX_BATCH_TOKENS` | tokens one `/v1/systemone/batch` FORWARD PASS may collate (`states` x questions x row width); a larger batch is split into several passes, not refused | `131072` |
| `LAYA_JEV_STRICT` | select System One response fields with no per-answer `action` / `answer_confidence`. Set to `0` for the extended Laya text response. Image requests always use the System One response format | `1` for single requests, `0` for batch |
| `LAYA_TOURNAMENT_GROUP` | answer a `choice` with more options than this by elimination, in groups of this size, on `/v1/systemone` and `/v1/systemone/batch`, and raise their option cap from 100 to 255. See [Large choices](#large-choices-with-laya_tournament_group). Unset, or outside 2 to 100, is off | `0` |

For a deployment published under a prefix such as `/laya`, set `LAYA_ROOT_PATH=/laya`.
FastAPI uses it when generating OpenAPI and Swagger UI URLs. Configure the reverse proxy to
strip `/laya` before forwarding requests to Laya; the app's routes remain `/health` and
`/v1/systemone`, `/v1/systemone/batch` and `/v1/route` internally.

For containers, including CUDA and ARM64 images, see [Docker quickstart](docker.md).

For bursty local use, set `LAYA_IDLE_UNLOAD_SECONDS=300`. Inference and unload run on the same
worker, and the idle window starts again when a single or batch forward pass finishes, including
failed requests. The next prediction pays a cold load. Unloading releases model references and
device caches, including Metal; the process allocator may retain RAM pages, so process RSS need
not fall by the size of the checkpoint.

## Endpoints

The default checkpoint answers text on `/v1/systemone` and native media on `/v1/multimodal`.
Set `LAYA_MULTIMODAL_MODEL` to a local trained checkpoint. Use Gemma 4 for audio.

The extended decision response includes `gate` details when a `min_confidence` policy
is applied. It records the threshold and the decision on whether to abstain.


### `GET /health`

Liveness is always open (no auth), and stays responsive during inference because the CPU-bound
forward pass runs on its own worker, not the event loop. The fields below liveness are not open on
a deployment that set `LAYA_API_KEY`: without the bearer, `/health` answers `{"status": "ok"}` and
nothing else, because the rest names resident checkpoints, their exact revision SHAs, the device
state and each checkpoint's last fallback reason, which quotes host hardware. A probe needs only
the 200, so a healthcheck is unaffected and a wrong bearer is still a 200 rather than a 401. With
no `LAYA_API_KEY` set, every caller gets the full payload shown here.

```json
{"status": "ok", "loaded": ["multimodal"], "revisions": {"multimodal": "..."},
 "device": "cuda", "device_is_preference": false,
 "checkpoint_devices": {"multimodal": "cuda"},
 "cpu_fallbacks": {"multimodal": {"count": 0, "last_reason": null}}}
```

One server's answer, so the blocks agree with each other: every key of `revisions`,
`checkpoint_devices` and `cpu_fallbacks` is a name in `loaded`. `tests/test_serve.py` holds this
sample to the handler that produces it, field by field.

With idle unloading enabled, authenticated health responses also include `idle_unload_seconds`
(the configured window) and `idle_seconds` (time since the last inference request or completion).
Health probes and routing snapshots do not reset that clock. An empty `loaded` list is normal
after an idle unload.

- `status` is `ok` whenever the process answers at all. It says nothing about the checkpoints.
- `loaded` lists the checkpoints resident in memory. It is empty until a request builds one, which is
  what `LAYA_PRELOAD=0` leaves the process doing.
- `revisions` is the artifact revision each resident checkpoint was loaded from, keyed by the same
  names as `loaded`, so a deployment can confirm what it is actually serving.
- `device` reports the device of a resident checkpoint. With no checkpoint loaded, it
  reports the configured preference. Use `LAYA_DEVICE=auto` to select an available device.
  An explicitly requested CUDA device must be available when the native checkpoint loads.
- `device_is_preference` is `true` while the service is reporting a preference and
  `false` when it can report a resident checkpoint's device.
- `checkpoint_devices` gives the measurement per checkpoint, keyed by the names in `loaded`;
  `device` is the first of those values.
- `cpu_fallbacks` exposes retry counters from backends that report them. The native
  multimodal backend reports a count of zero and a null reason.

### `POST /v1/route`

Inspect the deployed Router's checkpoint selection without requesting inference. This is useful
when an application's local routing configuration differs from the service's. MCP remote routing
still runs locally; it does not call this endpoint.

```bash
curl -s localhost:8000/v1/route -H 'content-type: application/json' -d '{
  "state": {"message": "Turn lights off"},
  "questions": {},
  "explain": true
}'
```

With the default fallback model, this returns:

```json
{"model": "multimodal", "repo": "ZeeoRe/jev-multimodal-qwen3.5-4b",
 "reason": "default native checkpoint",
 "diagnostic": {"stage": "whole_state", "reason": "insufficient_words", "source": null}}
```

`state` and `questions` are required, with the same shape and size limits as prediction. An empty
questions object is allowed. This is a routing snapshot, not a complete inference-request validator:
a successful route does not certify that every question will pass inference-time validation.

The optional routing controls are `model`, `task`, `lang` and `lang_guess`, with the same
[model aliases, registry and override semantics](#post-v1systemone) as prediction. Omitted controls
(and `null` routing overrides) leave deployment settings in charge. `explain` defaults to `false`;
only JSON booleans are accepted. Strings, numbers and `null` return `422` naming `explain` before
routing. Unknown fields are ignored, as on the prediction endpoints, but prediction-only controls
`max_len`, `head_max_len`, `min_confidence`, `batch_size`, `sort_by_length`, `diagnostics` and `states`
are refused with `422`, including when their value is `null`. The same applies to `hooks`,
`on_route`, `on_predict_start`, `on_predict_end`, `hooks_raise` and `hooks_timeout`: hook policy
belongs to the operator.

The default response contains only `model`, `repo` and `reason`. With `explain: true`, it also
contains `diagnostic`, read from the final decision **after** `on_route`; no detector is run a
second time. It is `null` when that decision carries no diagnostic. Built-in diagnostics contain
`stage`, `reason` and `source`, preserving `source: null` when no source is available. The full
`detection` mapping, including `mixed_segment` containing verbatim input, is never exported.
Built-in diagnostic metadata does not include caller values, but `source.path` may contain
caller-chosen field names. Hook-provided diagnostics remain operator-controlled. The routing
`reason` can include input text, so the response as a whole is **not text-free**.

Authentication, admission, body-read timeout and request limits are shared with prediction.
Routing runs behind the same gate on the same worker, so it can wait behind a forward pass;
use `/health` for liveness. The endpoint does not call prediction or load a checkpoint. Startup
preload is unchanged, and operator-installed hooks still run and may have side effects.
Cancellation of an HTTP request or a hook timeout does not forcibly stop code already running.
Routing does not reset the inference idle deadline on success, failure or cancellation, though
queued/running work can delay unloading. The result describes routing **now**, not a guarantee
about a later prediction if configuration, hooks or input change.

An injected legacy router without the `explain` parameter still supports omitted/false requests;
`explain: true` returns a clear `422`, without silently retrying. As with prediction, a `ValueError`
returns `422` with its message, including one raised by an operator hook. Other unexpected errors
are logged server-side and return `500` with `routing failed`.

#### Try a configuration mismatch locally

From a checkout, run the complete example without downloading checkpoints or starting a server:

```bash
python -m pip install -e ".[serve]" httpx
python examples/http_route.py
```

The local Router uses `multimodal`, while the injected server Router defaults to a registered
`support` checkpoint. The example checks both responses, an explicit checkpoint pin and
the empty resident model lists.

### `POST /v1/systemone`

One request carries a `state` and any number of questions over it. Those questions share that
state: each is answered against the whole serialized context, so a question whose instructions
name `rows[i]` still sees every record. One answer per record is one entry of `states` on
`POST /v1/systemone/batch`.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, I want my money back",
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                             "other": "everything else"}},
    "urgency": {"type": "score",  "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry", "furious"]}
  }
}'
```

| field | required | meaning |
|---|---|---|
| `state` | yes | the one context every question in the request is answered against: text, email, ticket or JSON; a missing or `null` state is a `400` |
| `questions` | yes | object keyed by question ID. Use `choice`, `score`, `noul`, or `box` with uploaded images |
| `model` | no | names a checkpoint; a path or unpublished Hub id is a `422`, anything else is ignored (see below) |
| `assets` | no | image reference map with inline PNG/JPEG data or multipart file part names |
| `task` | no | select a registered checkpoint by name when `model` is omitted |
| `lang` | no | optional language metadata for an attached engine |
| `lang_guess` | no | optional language hint for routing diagnostics |
| `max_len` | no | total token window for this request, capped by `LAYA_MAX_TOKEN_BUDGET` |
| `head_max_len` | no | reserved for backends that support a separate option budget. Native checkpoints use `max_len` |
| `min_confidence` | no | abstention threshold: a number in `[0.0, 1.0]`, or a per-bucket map keyed by option-count bucket (`{"choice:3-5": 0.9, "default": 0.4}`) so the threshold can differ by option count; an answer whose `answer_confidence` falls below its own threshold comes back marked `low_confidence`, and the answer itself is kept |

`model`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len` and `min_confidence` are the
arguments `Router.predict` takes that a JSON body can state; each is forwarded only when the request
sends it, so an absent one leaves the deployment's own `Router(...)` setting in charge. The five
hook arguments `predict` also takes -- `hooks`, `on_predict_start`, `on_predict_end`,
`hooks_raise`, `hooks_timeout` -- are refused with a `422` rather than dropped: a hook is a callable
that runs inside the server process, and the last two say how the hooks a deployment installed
execute, so no value a caller sends has a meaning here. The same five are refused client-side by a
LangChain node with a `base_url` (`laya.integrations.langchain`), so a chain and a raw HTTP client
now get the same answer. Upload images to System One with `assets`. Requests with
inline `image`, `audio` or `video` fields go to `/v1/multimodal`.

Use `multimodal`, `laya`, `default`, the released checkpoint ID
`ZeeoRe/jev-multimodal-qwen3.5-4b`, or a name registered in `LAYA_EXTRA_MODELS`.
Client model IDs such as `jev-1` select the server's default. Arbitrary paths and
unregistered Hub repo IDs return HTTP 422. Configure checkpoint sources on the service.

### Image assets and single-box questions

Send `TypeSafe-Danger-Zone-Images: true` with image or box requests. Both sides of each
PNG or JPEG must be between 280 and 1,568 pixels. Put each image
under a reference in `assets` and use the reference in a string value in `state`,
`instructions` or `criteria`. Reusing the reference places the image at each occurrence.
Use an array of image references followed by a text value to place the images before
the text in the native model prompt.
A `box` question needs one distinct referenced image. The response carries `left`, `top`,
`right` and `bottom` in `[0, 1]` and localization confidence.

~~~bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"multimodal","state":{"photo":"scene-ref"},"assets":{"scene-ref":{"type":"image","part":"scene"}},"questions":{"target":{"type":"box","instructions":"Locate the red mug."}}};type=application/json' \
  -F 'scene=@scene.jpg;type=image/jpeg'
~~~

For JSON uploads, give the asset `format` and `data_base64` instead of `part`. The
first multipart part is the JSON request and every file part belongs to one asset. Images
must have sides between 280 and 1568 pixels and files may be at most 16 MiB. The image
request body and expanded RGB image buffers each have a 32 MiB limit. The configured
checkpoint's token budget also applies. [TypeSafe Images](https://docs.typesafe.ai/images)

The Python and TypeScript SDKs accept JSON image assets and set the image header on the
request. [SDK examples](multimodal-clients.md#system-one-image-assets)

### Response

The default single-request response has `model`, `answers` and `usage`. Here is an
example using the bundled billing request and a native Qwen checkpoint.

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9967,
      "probabilities": {
        "billing": 0.9978,
        "technical": 0.0018,
        "sales": 0.0004
      }
    },
    "urgency": {
      "type": "score",
      "score": 0.9502,
      "confidence": 0.5125,
      "probabilities": {
        "0": 0.1874,
        "1": 0.675,
        "2": 0.1376
      },
      "legend": {
        "0": "not urgent",
        "1": "needs attention soon",
        "2": "critical deadline or blocking issue"
      }
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.998
    }
  },
  "usage": {
    "input_tokens": 286,
    "output_tokens": 0
  }
}
```

Each answer has the same ID as its question. A `choice` reports the selected option and
its distribution. A `score` reports the expected level index and a legend. A `noul`
reports the probability of yes. A `box` reports normalized coordinates and localization
confidence. Option probabilities are rounded to four decimals. Allow rounding tolerance
when checking their sum. [Probability precision](questions-and-answers.md#probability-precision)

`input_tokens` counts the prepared prompts across the executed questions. Native decisions
report `output_tokens` as zero. In extended responses, `usage.forward_passes` counts the calls to the backbone,
and `routing` identifies the checkpoint that answered. `/v1/route` with `explain: true`
can report language diagnostics.

The extended `gate` fields appear when `min_confidence` is set. `abstention` is
`passed`, `abstained` or `unevaluated`. The response also includes
`abstention_threshold` and marks low-confidence answers with `low_confidence: true`.
The `answer_confidence` used by the gate is the largest decision probability.

### Confidence: three numbers, not interchangeable

- `answer_confidence` is the probability mass on the reported answer (`max(p)`). It is the
  quantity temperature scaling fits and the one this repo's ECE figures are computed on, so it
  carries the gating property the [Benchmarks and known limits](benchmarks.md) page relies on --
  but only for a checkpoint whose temperature fit has been validated on your traffic.
- `confidence` means something different per type: normalized entropy `1 - H(p)/log(k)` on
  `choice` and `score`, and `max(p_yes, p_no)` on `noul` (where it equals `answer_confidence`).
- `x_jev_confidence` is the `confidence` Jev would report for the same probabilities, on `choice`
  and `score` answers only (a Jev `noul` answer has no confidence). It uses TypeSafe's own
  formulas: `(p_max - 1/n)/(1 - 1/n)` for `choice`, and for `score` one minus the probability-weighted
  distance from the most likely level, scaled by the same distance for a uniform distribution.
  The formula is the library's own, not the server's: `laya.jev_confidence(answer)` takes one
  answer dict -- the shape this route returns and the shape `Agent.predict` returns -- and gives
  the unrounded number, so a caller running Laya in-process filters on the same quantity the
  served field carries without an HTTP round trip.

Never compare them against one threshold. When porting from Jev, a threshold carried over from a
Jev deployment gates differently on Laya's entropy `confidence` (#302); read it against
`x_jev_confidence` instead. The `x_` prefix keeps the field clear of any name Jev adds later.

### Strict Jev contract: `LAYA_JEV_STRICT`

Single requests default to `model`, `answers` and `usage` with the two token counts.

- a `choice` answer keeps `type`, `choice`, `probabilities` and `confidence`
- a `score` answer keeps `type`, `score`, `probabilities`, `confidence` and `legend`
- a `noul` answer keeps `type` and `noul`
- a `box` answer keeps `type`, `box` and `confidence`

Choice and score `confidence` use the TypeSafe formulas applied to the returned probabilities.
A score's `legend` preserves structured criteria and original image references.
The server logs text truncation. Use the extended text response for in-band routing and
truncation details. The batch extension defaults to the full response and its optional strict
projection retains the batch's Laya confidence values.

Successful responses also carry `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms`.

### Large choices with `LAYA_TOURNAMENT_GROUP`

Set a group size from 2 to 100 to score large choices by elimination. Each round passes its
group winners to the next round. Extended responses include `x_tournament.finalists` and
`x_tournament.rounds`. Eliminated options have zero probability.

The tournament acts on each state in a batch and sums the token usage across the rounds.
When narrowing needs to discard a supplied option order, the extension includes
`option_order_dropped`. Measure its effect on your own checkpoint and labelled split.
[Tournament evaluation](evals.md)

## Limits

| Limit | Value |
| --- | --- |
| Standard JSON body | 2 MiB |
| System One image upload body | 32 MiB with the image header enabled |
| Individual System One image file | 16 MiB |
| Serialized text state | 50,000 characters |
| Questions per request | 64 |
| States per batch | 64 |
| Choice options per question | 100 for regular text requests. The image path, an explicit `model: "multimodal"` on single requests and tournament mode allow up to 255 |
| Score levels per question | 32 for text and 10 for System One image requests |
| Total answer options | 512 |
| Concurrent admitted requests | `LAYA_MAX_CONCURRENT`, default 16 |

The serving limits apply before model preprocessing. The checkpoint also limits the total
prompt length and the options its readout can score. Native requests use `max_len` for the
input budget. A prompt that exceeds it returns HTTP 422. Shorten the state, media or option
text to fit, or raise the request budget within `LAYA_MAX_TOKEN_BUDGET`.

`LAYA_MAX_BATCH_TOKENS` guides the HTTP batch chunk size. The native agent prepares each
state and packs compatible questions within its own token budget.
[Native model inputs](multimodal.md#local-prediction)

## Errors

| status | when | body `detail` |
|---|---|---|
| `400` | body is not valid JSON, not an object, has no `questions`, `state` is missing or `null`, `questions` is not an object, or a string anywhere in the body holds an unpaired `\udXXX` surrogate escape | what is wrong |
| `401` | `LAYA_API_KEY` is set and the bearer token is missing or wrong | `invalid or missing bearer token` |
| `413` | any limit above | which limit and by how much |
| `422` | the question is well-formed JSON but invalid to Laya (unknown type, options over the head budget), or a request control (`lang`, `min_confidence`, a hook argument) is not in the form this endpoint accepts | names the question or the field and what to fix |
| `500` | inference failed for any other reason | `inference failed` -- always this string, so paths, weights and memory state never leak; the cause is in the server log |
| `503` | `LAYA_MAX_CONCURRENT` requests are already in flight | `server busy, try again later` |

The unpaired-surrogate `400` is the one that looks unusual. `\udXXX` with no pair is legal JSON, but
the character it names cannot be UTF-8 encoded, so the tokenizer raises `TypeError` on it -- the
caller's own string arriving as a server fault, with a traceback per request. Both decision routes
therefore walk the parsed body for lone surrogates and refuse one before it reaches inference. The
walk runs after the size checks, so an oversized body is still refused first and the character and
question limits bound what it can reach. A *paired* surrogate is one ordinary astral character by
the time the parser is done, so an emoji in a state is unaffected.

Over-cap load is refused, not queued: clients holding an admission slot while streaming a slow body
cannot starve `/health`, and a retry can take the slot a refused client left.

## Concurrency model

Inference is a synchronous torch call that takes hundreds of milliseconds to seconds on CPU, so it
never runs on the event loop: requests are handed to a single-worker executor, which means one
forward pass at a time -- the shape a single checkpoint on one device wants. Admission (the
`LAYA_MAX_CONCURRENT` semaphore) is checked before any body byte is read and held through
inference; the inference gate is joined only after the body is complete, so a slow client holds an
admission slot but never an inference slot. That ordering is what bounds how many bodies are
buffered at once, so the body read carries its own deadline (`LAYA_BODY_READ_TIMEOUT`, 10 s): a peer
that sends a request head and then stops writing answers `408` and gives the slot back, rather than
holding one of the `LAYA_MAX_CONCURRENT` slots -- and so both decision routes -- out of service for
as long as it keeps the socket open.

## Batch and media requests

`POST /v1/systemone/batch` accepts an array of `states` and one shared `questions` object.
It returns `results` in the input order and sums the token counts in `total_usage`.
Use `POST /v1/multimodal` for an inline audio or video file. Named PNG and JPEG assets for
text or single-box decisions use `POST /v1/systemone`.
[Python, TypeScript and curl examples](multimodal-clients.md)
