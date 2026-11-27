using System.Net.Http.Headers;

if (args.Length != 2)
{
    Console.Error.WriteLine("Usage: dotnet run --project dotnet -- ENDPOINT REQUEST.json");
    return 2;
}

using var client = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false })
{
    Timeout = TimeSpan.FromSeconds(120),
};
using var payload = File.OpenRead(args[1]);
using var request = new HttpRequestMessage(HttpMethod.Post, args[0])
{
    Content = new StreamContent(payload),
};
request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
var key = Environment.GetEnvironmentVariable("LAYA_API_KEY");
if (!string.IsNullOrWhiteSpace(key))
{
    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
}
using var response = await client.SendAsync(request);
response.EnsureSuccessStatusCode();
Console.WriteLine(await response.Content.ReadAsStringAsync());
return 0;
