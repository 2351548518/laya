# laya-multimodal

Local System 1 decisions over text and native media. Each question uses one forward pass
and returns a `choice`, `score`, `noul` or a single `box`.

A `choice` question picks one of the supplied options, a `score` question gives a distribution
over ordered levels, and a `noul` question gives the probability of yes. A `box` question localizes
one referred object in an image.

## Start here

Install the source checkout, download the released checkpoint and run the first prediction.
[Installation and quickstart](https://github.com/2351548518/laya#quickstart)

## Find a guide

| To | Read |
|---|---|
| run or train native text, image, audio and video decisions | [Native multimodal decisions](multimodal.md) |
| inspect the v6 architecture, tensor dimensions and training examples in Chinese | [v6 project guide](project-readme/index.md) |
| try native prediction and inspect example training records | [Runnable examples](https://github.com/2351548518/laya/tree/main/examples) and the [native Notebook](https://github.com/2351548518/laya/blob/main/notebooks/laya_native_multimodal.ipynb) |
| send media to your local server from Python or TypeScript | [Native media SDKs](multimodal-clients.md) |
| predict from a terminal with a trained local model | [Native multimodal CLI](cli-mcp.md#native-multimodal-checkpoint) |
| locate one object in an image | [Single-box grounding](grounding.md) |
| download the released model or deploy a quantized variant locally | [Models and GGUF deployment](multimodal-release.md) |
| get typed values back from a JSON schema or a pydantic model | [Schema-driven decisions](structured.md) |
| log, redact, cache or gate every decision without forking Laya | [Prediction hooks](hooks/index.md) |
| adopt Laya incrementally without granting execution permission | [Staged adoption](staged-adoption.md) |
| route, screen or triage inside a LangChain or LangGraph app | [LangChain & LangGraph](langchain.md) |
| run Laya from a terminal or connect its built-in MCP stdio server | [Command line and MCP server](cli-mcp.md) |
| hand each request to the checkpoint that fits its language and task | [Routing](routing.md) |
| call Laya over HTTP from a client written against the Jev wire protocol | [HTTP API](http-api.md) |
| run the SDK or the `laya-serve` HTTP API in a container, on CPU or an NVIDIA GPU | [Docker quickstart](docker.md) |
| build for ARM64 hosts or DGX Spark | [ARM64 and DGX Spark containers](docker-platforms.md) |
| specialise a checkpoint for your own decisions | [Browser-agent fine-tuning example](finetune_browser_agent.md) and the [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) |
| judge benchmark results and deployment limits | [Benchmarks and known limits](benchmarks.md) |
| look up a class, function or parameter | [Python API reference](reference/index.md) |
| score a native decision or grounding dataset | [Native benchmarks](benchmarks.md#native-multimodal-checkpoints) |
| gate a build on a labelled classic text dataset | [Classic text evaluation harness](evals.md) |

See the inherited text model guide for CLI flags, batch hooks, confidence gating and Vega.
[Laya text routing and integrations](laya-guide.md)
