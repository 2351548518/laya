import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;

class DecisionClient {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: java DecisionClient.java ENDPOINT REQUEST.json");
        }
        var builder = HttpRequest.newBuilder(URI.create(args[0]))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofFile(Path.of(args[1])));
        String key = System.getenv("LAYA_API_KEY");
        if (key != null && !key.isBlank()) {
            builder.header("Authorization", "Bearer " + key);
        }
        var response = HttpClient.newHttpClient()
                .send(builder.build(), HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() / 100 != 2) {
            throw new IllegalStateException("Decision request returned HTTP " + response.statusCode());
        }
        System.out.println(response.body());
    }
}
