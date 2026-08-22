#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

std::chrono::steady_clock::time_point program_start;
double now_ms() {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - program_start).count();
    return us/1000.0;
}

struct Result {
    int    id;
    int    prompt_length;
    int    true_out_len;
    double arrival_ms;
    double start_ms;
    double first_token_ms;
    double end_ms;
};

struct Request {
    int id;
    int prompt_length;
    int max_tokens;
    double arrival_ms;
    std::promise<Result> promise;
};

//EngineConfig is basically parameters for a mock GPU. here we just 
//simplify hardware physics to test the concepts

struct EngineConfig {
    double p0 = 5.0; // prefill base (ms)
    double p1 = 0.1; // prefill per prompt token (ms)
    double d0 = 10.0; // decode base per step (ms)
    double d1 = 0.5; // decode extra per seat (ms)
    int max_tokens = 512;
    int max_slots = 4; // seats per batch
    double batch_timeout_ms = 50;  
    bool pace_with_sleeps = true; 
    bool continuous = true;
    unsigned seed = 67;
};

struct Slot {
    Request request;
    int true_out_len = 0;
    int tokens_generated = 0;
    double start_ms = 0.0;
    double first_token_ms = 0.0;
    double end_ms = 0.0;
    int finish_tick = -1;
    bool finished= false;
};

template<typename T>
class ThreadSafeQueue{
    public:
      void push(T item){
        {
        std::lock_guard<std::mutex> lock(mtx_);
        q_.push(std::move(item));  
        } 
        cv_.notify_one();
      }

      T pop(){
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]{return !q_.empty();});
        T item=std::move(q_.front());
        q_.pop();
        return item;
      }

    std::optional<T> try_pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, timeout, [this] { return !q_.empty(); }))
            return std::nullopt;
        T item = std::move(q_.front());
        q_.pop();
        return item;
    }

    int size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return (int)q_.size();
    }

private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

int sample_out_length(std::mt19937& rng) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    if (dist(rng) < 0.8) {
        std::uniform_int_distribution<> short_len(10, 50);
        return short_len(rng);
    } else {
        std::uniform_int_distribution<> long_len(100, 500);
        return long_len(rng);
    }
}

struct AdmissionPolicy {
    virtual std::vector<int> pick(const std::vector<Request>& waiting,int free_slots)= 0;
    virtual ~AdmissionPolicy()=default;
};

struct FCFSPolicy: AdmissionPolicy {
    std::vector<int> pick(const std::vector<Request>& waiting,int free_slots) override {
        std::vector<int> idx;
        for (int i=0;i<(int)waiting.size()&&(int)idx.size()<free_slots;i++)
            idx.push_back(i);
        return idx;
    }
};

struct ShortPromptFirstPolicy: AdmissionPolicy {
    std::vector<int> pick(const std::vector<Request>& waiting, int free_slots) override {
        std::vector<int> idx;
        for (int i=0;i<(int)waiting.size();i++) idx.push_back(i);
        std::sort(idx.begin(),idx.end(), [&](int a,int b) {   
            return waiting[a].prompt_length < waiting[b].prompt_length;  
        });                                                     
        if((int)idx.size()>free_slots) idx.resize(free_slots);
        return idx;
    }
};

void worker_static(ThreadSafeQueue<Request>& q, const EngineConfig& config, AdmissionPolicy& policy) {
    std::mt19937 rng(config.seed);
    std::vector<Request> waiting;   
    bool done = false;              

    long total_idle = 0, total_seat_ticks = 0;
    int total_batches = 0;

    while (!(done && waiting.empty())) {
        if (waiting.empty()) {
            Request first = q.pop();                    // nap until someone arrives
            if (first.id == -1) done = true;
            else waiting.push_back(std::move(first));
        }
        while (!done && (int)waiting.size() < config.max_slots) {
            auto m = q.try_pop_for(std::chrono::milliseconds((long)config.batch_timeout_ms));
            if (!m) break;
            if (m->id == -1) { done = true; break; }
            waiting.push_back(std::move(*m));
        }

        std::vector<int> picks = policy.pick(waiting, config.max_slots);

        std::sort(picks.begin(), picks.end(), [](int a, int b) { return a > b; }); // back->front
        std::vector<Slot> batch;
        for (int idx: picks) {
            Slot s;
            s.request = std::move(waiting[idx]);
            batch.push_back(std::move(s));
            waiting.erase(waiting.begin() + idx);       //so indices don't shift
        }
        if (batch.empty()) continue;

        //run static batch
        int seats=(int)batch.size();
        double step_ms = config.d0 + (config.d1 * seats);
                                                       
        double batch_start = now_ms();
        for (auto& s : batch) {
            s.true_out_len= sample_out_length(rng);
            s.tokens_generated= 1;                   
            s.start_ms= batch_start;
        }
        double prefill = 0;
        for (auto& s : batch)
            prefill = std::max(prefill, config.p0 + (config.p1 * s.request.prompt_length));
        double t= batch_start + prefill;              
        for (auto& s : batch) s.first_token_ms = t;

        int finished = 0, ticks = 0;
        while (finished < seats) {
            t += step_ms;                               // one decode tick and every seat pays
            ++ticks;
            for (auto& s : batch) {
                if (s.finished) continue;              
                ++s.tokens_generated;
                if (s.tokens_generated >= s.true_out_len) {
                    s.finished    = true;
                    s.finish_tick = ticks;
                    s.end_ms      = t;
                    ++finished;
                    s.request.promise.set_value(Result{   
                        s.request.id, s.request.prompt_length,     
                        s.true_out_len, s.request.arrival_ms,       
                        s.start_ms, s.first_token_ms, s.end_ms});
                }
            }
        }

        long idle = 0;
        for (auto& s : batch) idle += ticks - s.finish_tick;
        total_idle+= idle;
        total_seat_ticks += (long)seats * ticks;
        ++total_batches;

        if (config.pace_with_sleeps)
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(t - batch_start));
    }

    std::cerr << "[static] batches=" << total_batches
              << " idle_seat_ticks=" << total_idle << "/" << total_seat_ticks
              << " (" << (total_seat_ticks > 0 ? 100.0 * total_idle / total_seat_ticks : 0.0)
              << "% wasted)\n";
}

void worker_continuous(ThreadSafeQueue<Request>& q, const EngineConfig& cfg,
                       AdmissionPolicy& policy) {
    std::mt19937 rng(cfg.seed);
    std::vector<Request> waiting;    
    bool done= false;
    std::vector<Slot> active;                                             
    int total_ticks= 0;
    long tokens_made= 0;
    std::vector<int> occupancy;        
    double t = now_ms();             
    double run_start = -1;           

    while (!(done && waiting.empty() && active.empty())) {
        if (active.empty() && waiting.empty() && !done) {
            Request first = q.pop();              
            if (first.id == -1) done = true;      
            else waiting.push_back(std::move(first));
        }
        while (true) {                              
            auto m = q.try_pop_for(std::chrono::milliseconds(0));
            if (!m) break;
            if (m->id == -1) { done = true; break; }   
            waiting.push_back(std::move(*m));
        }
 
        for (int i = (int)active.size() - 1; i >= 0; --i)
            if (active[i].finished)
                active.erase(active.begin() + i);

        int free_slots = cfg.max_slots - (int)active.size();
        if (free_slots> 0 && !waiting.empty()) {
            std::vector<int> picks = policy.pick(waiting, free_slots);
            std::sort(picks.begin(), picks.end(), [](int a, int b) { return a > b; });
            for (int idx: picks) {
                Slot s;
                s.request= std::move(waiting[idx]);
                s.true_out_len = sample_out_length(rng);
                s.start_ms= t;
                s.first_token_ms= t + cfg.p0 + cfg.p1 * s.request.prompt_length;
                s.tokens_generated = 1;             // prefills free first token
                if (run_start < 0) run_start = t;
                tokens_made += 1;
                active.push_back(std::move(s));
                waiting.erase(waiting.begin() + idx);
            }
        }

        if (active.empty()) continue;               
      
        double step_ms = cfg.d0 + cfg.d1 * (int)active.size();  
        t += step_ms;                                            
        total_ticks++;
        occupancy.push_back((int)active.size());

        for (auto& s : active) {
            ++s.tokens_generated;
            ++tokens_made;
            if (s.tokens_generated >= s.true_out_len) {
                s.finished = true;                 
                s.end_ms   = t;                    
                s.request.promise.set_value(Result{
                    s.request.id, s.request.prompt_length, s.true_out_len,
                    s.request.arrival_ms, s.start_ms, s.first_token_ms, s.end_ms});
            }
        }

        if (cfg.pace_with_sleeps)
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(step_ms));
    }

    long idle_ticks = 0;
    for (int k = 0; k < (int)occupancy.size(); k++)
        idle_ticks += cfg.max_slots - occupancy[k];
    double avg_occupancy = occupancy.empty() ? 0.0
        : (double)std::accumulate(occupancy.begin(), occupancy.end(), 0) / occupancy.size();
    double duration = t - (run_start >= 0 ? run_start : t);
    double throughput = duration > 0 ? tokens_made / duration * 1000.0 : 0.0;

    std::cerr << "[continuous] ticks=" << total_ticks << " tokens=" << tokens_made
              << " duration_ms=" << duration << " throughput_tok_s=" << throughput
              << " avg_occupancy=" << avg_occupancy << "/" << cfg.max_slots
              << " idle_seat_ticks=" << idle_ticks << "\n";
}

struct TraceEntry {
    double arrival_ms;  // simulated arrival time from t=0
    int prompt_length;
};

std::vector<TraceEntry> generate_trace(unsigned seed, int n, double lambda) {
    std::mt19937 rng(seed);
    std::exponential_distribution<> gap_s(lambda);
    std::uniform_int_distribution<> prompt_len(10, 500);

    std::vector<TraceEntry> trace;
    trace.reserve(n);
    double t = 0.0;
    for (int i = 0; i < n; ++i) {
        t += gap_s(rng) * 1000.0;
        trace.push_back({ t, prompt_len(rng) });
    }
    return trace;
}

struct CliOptions {
    std::string mode = "continuous";
    std::string policy = "fcfs";
    double lambda = 5.0;
    unsigned seed = 42;
    std::string out = "results.csv";
    int n = 200;
    bool pace = true;
    bool verbose = false;
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions opt;
    if (argc == 8) {
        opt.mode   = argv[1];
        opt.policy = argv[2];
        opt.lambda = std::stod(argv[3]);
        opt.seed   = std::stoi(argv[4]);
        opt.out    = argv[5];
        opt.n      = std::stoi(argv[6]);
        opt.pace   = (std::string(argv[7]) == "1");
    } else if (argc > 1) {
        std::cerr << "warning: expected 7 arguments, using defaults\n";
    }
    
    return opt;
}

int main(int argc, char** argv) {
    program_start = std::chrono::steady_clock::now();
    CliOptions opt = parse_args(argc, argv);

    EngineConfig config;
    if (opt.mode == "continuous")      config.continuous = true;
    else if (opt.mode == "static")     config.continuous = false;
    else {
        std::cerr << "warning: unknown --mode '" << opt.mode << "', using continuous\n";
        config.continuous = true;
    }
    config.seed             = opt.seed;
    config.pace_with_sleeps = opt.pace;

    std::unique_ptr<AdmissionPolicy> policy;
    if (opt.policy == "shortprompt") {
        policy = std::make_unique<ShortPromptFirstPolicy>();
    } else {
        if (opt.policy != "fcfs")
            std::cerr << "warning: unknown --policy '" << opt.policy << "', using fcfs\n";
        policy = std::make_unique<FCFSPolicy>();
    }

    ThreadSafeQueue<Request> queue;
    std::thread worker_thread(config.continuous ? worker_continuous : worker_static,
                              std::ref(queue), std::ref(config), std::ref(*policy));

    auto trace = generate_trace(opt.seed, opt.n, opt.lambda);

    std::vector<std::future<Result>> futures;
    futures.reserve(opt.n);
    for (int i = 0; i < opt.n; ++i) {
        Request req;
        req.id = i;
        req.prompt_length = trace[i].prompt_length;
        req.max_tokens = config.max_tokens;
        if (opt.pace) {
            double now = now_ms();
            if (trace[i].arrival_ms > now)
                std::this_thread::sleep_for(
                    std::chrono::duration<double, std::milli>(trace[i].arrival_ms - now));
            req.arrival_ms = now_ms();
        } else {
            req.arrival_ms = trace[i].arrival_ms;
        }
        futures.push_back(req.promise.get_future());
        queue.push(std::move(req));
    }

    Request sentinel; sentinel.id = -1;
    queue.push(std::move(sentinel));

    std::vector<Result> results;
    results.reserve(opt.n);
    for (auto& f : futures) results.push_back(f.get());
    worker_thread.join();

    std::ofstream out(opt.out);
    if (!out) {
        std::cerr << "error: could not open '" << opt.out << "' for writing\n";
        return 1;
    }
    out << std::fixed << std::setprecision(3);
    out << "mode,policy,lambda,seed,id,prompt_length,true_out_len,"
           "arrival_ms,start_ms,first_token_ms,end_ms\n";
    for (auto& r : results) {
        out << opt.mode << ',' << opt.policy << ',' << opt.lambda << ',' << opt.seed << ','
            << r.id << ',' << r.prompt_length << ',' << r.true_out_len << ','
            << r.arrival_ms << ',' << r.start_ms << ',' << r.first_token_ms << ',' << r.end_ms
            << '\n';

        if (opt.verbose) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "req " << r.id
                      << " | wait "  << r.start_ms - r.arrival_ms
                      << " ms | TTFT " << r.first_token_ms - r.arrival_ms
                      << " ms | total " << r.end_ms - r.arrival_ms << " ms\n";
        }
    }
    std::cerr << "wrote " << results.size() << " rows to " << opt.out << "\n";
    return 0;
}
