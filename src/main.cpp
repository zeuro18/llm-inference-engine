#include <iostream>  
#include <thread>
#include <chrono>
#include <random>
#include <vector>
std::chrono::steady_clock::time_point program_start;
double now_ms() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - program_start).count();
    return us/1000.0;
}
struct Request{
    int id;
    int prompt_length;
    int max_tokens;
    double arrival_ms;
};
struct Result {
    int    id;
    int    prompt_length;
    int    true_out_len;    // hardcoded for now(50)
    double arrival_ms;
    double start_ms;
    double first_token_ms;
    double end_ms;
};

//EngineConfig is basically parameters for a mock GPU. here we just 
//simplify hardware physics to test the concepts

struct EngineConfig{
    double p0=5.0; //prefill base cost
    double p1=0.1; //prefill costper token
    double d0=10.0; //decode base cost
    double d1=0.5;  //decode extra cost(per batch member)
    //all in ms^
    int max_tokens=512;
    
};

int sample_out_length(std::mt19937& rng){
    std::uniform_real_distribution<> dist(0.0,1.0);
    if(dist(rng )<0.8){
        std::uniform_int_distribution<> short_len(10, 50);
        return short_len(rng);
    } else{
         std::uniform_int_distribution<> long_len(100, 500);
        return long_len(rng);
    }

    
}
Result kitchen(const Request& req, const EngineConfig& config, int batch_size, std::mt19937& rng){
    double start=now_ms();   
    double prefill_ms=config.p0+(config.p1* req.prompt_length); //reading=base+per-token cost
    double step_ms=config.d0+(config.d1* batch_size); //writing cost
    double first_token= start+ prefill_ms; 
    int N=sample_out_length(rng); 
    double end= first_token + (N-1) * step_ms;

    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(end - start));

    return Result{req.id, req.prompt_length, N,
                  req.arrival_ms, start, first_token, end};

}

int main() {
    program_start = std::chrono::steady_clock::now();
    std::mt19937 rng(0);        // fixed seed → same workload every run
    EngineConfig config;

    Result a = kitchen(Request{0, 50, 512, 0.0}, config, 1, rng);
    std::cout << "Test A (batch=1):\n";
    std::cout << "  TTFT: " << a.first_token_ms - a.arrival_ms << " ms\n";
    std::cout << "  TPOT: " << (a.end_ms - a.first_token_ms) / (a.true_out_len - 1)
              << " ms/token\n";
    std::cout << "  tokens/sec: " << a.true_out_len / (a.end_ms - a.start_ms) * 1000.0
              << "\n\n";

    std::vector<int> lengths;
    for (int i = 0; i < 16; ++i) lengths.push_back(sample_out_length(rng));
    int total_tokens = 0, max_N = 0;
    for (int N : lengths) { total_tokens += N; if (N > max_N) max_N = N; }

    double prefill = config.p0 + config.p1 * 50;                 // everyone has a 50-token prompt

    double t_one_by_one = 0;                               // scenario 1: sequential, batch=1
    for (int N : lengths)
        t_one_by_one += prefill + (N - 1) * (config.d0 + config.d1 * 1);

    double t_batch16 = prefill + (max_N - 1) * (config.d0 + config.d1 * 16);  // scenario 2: one static batch

    std::cout << "Test B: 16 requests, total " << total_tokens << " tokens\n";
    std::cout << "  per-token cost: one-by-one " << config.d0 + config.d1
              << " ms | batch16 " << config.d0 + config.d1 * 16 << " ms\n";
    std::cout << "  one-by-one: " << total_tokens / t_one_by_one * 1000.0 << " tokens/sec\n";
    std::cout << "  batch16:    " << total_tokens / t_batch16 * 1000.0 << " tokens/sec\n";
    std::cout << "  speedup:    " << t_one_by_one / t_batch16 << "x\n";
    return 0;
}