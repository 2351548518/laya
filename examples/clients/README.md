# Local HTTP examples

These examples send one JSON file to a running decision server. Use the same client
for text at `/v1/systemone` and for inline media at `/v1/multimodal`.
[Build a request](../../docs/multimodal-clients.md#http-without-an-sdk)

From this directory, select the endpoint and request file.

```bash
endpoint=http://127.0.0.1:8000/v1/systemone
request=../docker/request.json

go run main.go "$endpoint" "$request"
java DecisionClient.java "$endpoint" "$request"
dotnet run --project dotnet -- "$endpoint" "$request"
rustc --edition=2021 decision.rs -o /tmp/laya-rust-decision
/tmp/laya-rust-decision "$endpoint" "$request"
c++ -std=c++17 decision.cpp -lcurl -o /tmp/laya-cpp-decision
/tmp/laya-cpp-decision "$endpoint" "$request"
```

Use Go 1.16+, a JDK 11+, .NET 8+, Rust with curl on PATH, or a C++17 compiler with libcurl
development headers. Set `LAYA_API_KEY` when the server requires bearer authentication.
Use the maintained SDKs for typed request and response validation.
[Python client](../../sdk/python/README.md) · [TypeScript client](../../sdk/typescript/README.md)
