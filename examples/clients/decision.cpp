#include <curl/curl.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: decision ENDPOINT REQUEST.json\n";
        return 2;
    }
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) {
        std::cerr << "Could not open the request file\n";
        return 2;
    }
    const std::string payload((std::istreambuf_iterator<char>(file)), {});
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> client(curl_easy_init(), curl_easy_cleanup);
    if (!client) {
        return 1;
    }
    auto* headers = curl_slist_append(nullptr, "Content-Type: application/json");
    if (!headers) {
        return 1;
    }
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> owned(headers, curl_slist_free_all);
    if (const char* key = std::getenv("LAYA_API_KEY"); key && *key) {
        const auto auth = std::string("Authorization: Bearer ") + key;
        if (!curl_slist_append(headers, auth.c_str())) {
            return 1;
        }
    }
    curl_easy_setopt(client.get(), CURLOPT_URL, argv[1]);
    curl_easy_setopt(client.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(client.get(), CURLOPT_POSTFIELDS, payload.data());
    curl_easy_setopt(client.get(), CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(payload.size()));
    curl_easy_setopt(client.get(), CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(client.get(), CURLOPT_FAILONERROR, 1L);
    const CURLcode result = curl_easy_perform(client.get());
    if (result != CURLE_OK) {
        std::cerr << curl_easy_strerror(result) << '\n';
        return 1;
    }
    long status = 0;
    curl_easy_getinfo(client.get(), CURLINFO_RESPONSE_CODE, &status);
    return status / 100 == 2 ? 0 : 1;
}
