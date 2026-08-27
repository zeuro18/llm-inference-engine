#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

std::chrono::steady_clock::time_point program_start;

double now_ms() {
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - program_start).count();
  return us/1000.0;
}

struct Result {
  int id;
  int tier = 0; // 0=Enterprise, 1=Pro, 2=Free
  int prompt_length;
  int true_out_len;
  double arrival_ms;
  double start_ms;
  double first_token_ms;
  double end_ms;
};

struct Request {
  int id;
  int tier = 0;
  int prompt_length;
  int max_tokens;
  double arrival_ms;
  std::promise<Result> promise;
  double ttft_deadline_ms = 0.0;
};

//EngineConfig is basically parameters for a mock GPU. here we just
//simplify hardware physics to test the concepts
struct EngineConfig {
  double p0 = 5.0;   // prefill base (ms)
  double p1 = 0.1;   // prefill per prompt token (ms)
  double d0 = 10.0;  // decode base per step (ms)
  double d1 = 0.5;   // decode extra per seat (ms)
  int max_tokens = 512;
  int max_slots = 4;         
  double batch_timeout_ms = 50;
  bool pace_with_sleeps = true;
  bool continuous = true;
  unsigned seed = 67;
  int max_batch_tokens=500;
  int prefill_chunk_size=256;
  int kv_block_size=16;
  int total_kv_blocks=256;
  double ttft_slo_ms=500.0;
};

// aggregate stats by a worker for 1 run
struct RunStats {
  long total_batches = 0;
  long idle_seat_ticks = 0;
  long total_seat_ticks = 0;
  int total_ticks = 0;
  long tokens_made = 0;
  long total_preemptions = 0;
  double avg_occupancy = 0.0;
  double duration_ms = 0.0;
  double throughput_tok_s = 0.0;
  double utilization = 0.0; // fraction of ticks doing useful work
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
  int prompt_tokens_processed=0;
  bool prefill_done=false;
  int kv_blocks_used=0;
  bool preempted=false;
};

struct kvBlockAllocator{
  int total=0;
  int used=0;
  kvBlockAllocator(int total_blocks): total(total_blocks), used(0){}
  int free_blocks() const {return total - used;}
  int allocate(int n) {
    int can_give=std::min(n, free_blocks());
    used+=can_give;
    return can_give;
  }
  void release(int n) { used -= n; if(used<0) used=0; }
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

int slot_cost(const Slot& s, int chunk_size) {
  if (!s.prefill_done) {
    int remaining_prompt = s.request.prompt_length - s.prompt_tokens_processed;
    return std::min(chunk_size, remaining_prompt);
  }
  return 1; // 1 decode token
}

struct AdmissionPolicy {
  virtual std::vector<int> pick(
      const std::vector<Request>& waiting,
      int free_slots,
      int budget_remaining,
      int chunk_size,
      double current_time) = 0;
  virtual ~AdmissionPolicy() = default;
};

struct FCFSPolicy : AdmissionPolicy {
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      int free_slots,
      int budget_remaining,
      int chunk_size,
      double /*current_time*/) override {
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i = 0; i < (int)waiting.size() && (int)idx.size() < free_slots; i++) {
      int first_chunk_cost = std::min(waiting[i].prompt_length, chunk_size);
      if (first_chunk_cost <= budget) {
        idx.push_back(i);
        budget -= first_chunk_cost;
      }
    }
    return idx;
  }
};

struct ShortPromptFirstPolicy : AdmissionPolicy {
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      int free_slots,
      int budget_remaining,
      int chunk_size,
      double /*current_time*/) override {
    std::vector<int> order;
    for (int i = 0; i < (int)waiting.size(); i++) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return waiting[a].prompt_length < waiting[b].prompt_length;
    });
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i : order) {
      int first_chunk_cost = std::min(waiting[i].prompt_length, chunk_size);
      if (first_chunk_cost > budget) break;
      idx.push_back(i);
      budget -= first_chunk_cost;
      if ((int)idx.size() >= free_slots) break;
    }
    std::sort(idx.begin(), idx.end());
    return idx;
  }
};

struct SLOPolicy : AdmissionPolicy{
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      int free_slots,
      int budget_remaining,
      int chunk_size,
      double current_time) override {
   
    std::vector<int> order;
    for (int i = 0; i < (int)waiting.size(); i++) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      double slack_a = waiting[a].ttft_deadline_ms - current_time;
      double slack_b = waiting[b].ttft_deadline_ms - current_time;
      return slack_a < slack_b;  //smaller slack = more urgent = goes first
    });
 
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i : order) {
      if ((int)idx.size() >= free_slots) break;
      int cost = std::min(waiting[i].prompt_length, chunk_size);
      if (cost <= budget) {
        idx.push_back(i);
        budget -= cost;
      }
    }
    std::sort(idx.begin(), idx.end()); 
    return idx;
  }

};

void worker_static(ThreadSafeQueue<Request>& q, const EngineConfig& config,
                    AdmissionPolicy& policy, RunStats& stats) {
  std::mt19937 rng(config.seed);
  std::vector<Request> waiting;
  bool done = false;
  long total_idle = 0, total_seat_ticks = 0;
  int total_batches = 0;
  double sim_clock = 0.0;

  while (!(done && waiting.empty())) {
    if (waiting.empty()) {
      Request first = q.pop(); // nap until someone arrives
      if (first.id == -1) done = true;
      else waiting.push_back(std::move(first));
    }

    while (!done && (int)waiting.size() < config.max_slots) {
      auto m = q.try_pop_for(std::chrono::milliseconds((long)config.batch_timeout_ms));
      if (!m) break;
      if (m->id == -1) { done = true; break; }
      waiting.push_back(std::move(*m));
    }

    std::vector<int> picks = policy.pick(waiting, config.max_slots, config.max_batch_tokens, config.prefill_chunk_size, sim_clock);
    std::sort(picks.begin(), picks.end(), [](int a, int b) { return a > b; }); // back->front

    std::vector<Slot> batch;
    for (int idx : picks) {
      Slot s;
      s.request = std::move(waiting[idx]);
      batch.push_back(std::move(s));
      waiting.erase(waiting.begin() + idx); 
    }
    if (batch.empty()) continue;

    int seats = (int)batch.size();
    double step_ms = config.d0 + (config.d1 * seats);
    double max_arrival = 0.0;
    for (auto& s : batch) max_arrival = std::max(max_arrival, s.request.arrival_ms);
    double batch_start = std::max(sim_clock, max_arrival);

    for (auto& s : batch) {
      s.true_out_len = sample_out_length(rng);
      s.tokens_generated = 1;
      s.start_ms = batch_start;
    }

    double prefill = 0;
    for (auto& s : batch)
      prefill = std::max(prefill, config.p0 + (config.p1 * s.request.prompt_length));
    double t = batch_start + prefill;
    for (auto& s : batch) s.first_token_ms = t;

    int finished = 0, ticks = 0;
    while (finished < seats) {
      t += step_ms; 
      ++ticks;
      for (auto& s : batch) {
        if (s.finished) continue; 
        ++s.tokens_generated;
        if (s.tokens_generated >= s.true_out_len) {
          s.finished = true;
          s.finish_tick = ticks;
          s.end_ms = t;
          ++finished;
          s.request.promise.set_value(Result{ 
              s.request.id, s.request.tier, s.request.prompt_length, 
              s.true_out_len, s.request.arrival_ms, 
              s.start_ms, s.first_token_ms, s.end_ms});
        }
      }
    }

    long idle = 0;
    for (auto& s : batch) idle += ticks - s.finish_tick;
    total_idle += idle;
    total_seat_ticks += (long)seats * ticks;
    ++total_batches;
    sim_clock = t; // carry the simulated clock forward into the next batch

    if (config.pace_with_sleeps)
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(t - batch_start));
  }

  stats.total_batches = total_batches;
  stats.idle_seat_ticks = total_idle;
  stats.total_seat_ticks = total_seat_ticks;
  stats.utilization = total_seat_ticks > 0
      ? 1.0 - (double)total_idle / total_seat_ticks
      : 0.0;

  std::cerr << "[static] batches=" << total_batches
            << " idle_seat_ticks=" << total_idle << "/" << total_seat_ticks
            << " (" << (total_seat_ticks > 0 ? 100.0 * total_idle / total_seat_ticks : 0.0)
            << "% wasted)\n";
}

void worker_continuous(ThreadSafeQueue<Request>& q, const EngineConfig& cfg,
                        AdmissionPolicy& policy, RunStats& stats) {
  std::mt19937 rng(cfg.seed);
  std::vector<Request> waiting;
  bool done = false;
  std::vector<Slot> active;
  int total_ticks = 0;
  long tokens_made = 0;
  std::vector<int> occupancy;
  double t = 0.0;
  double run_start = -1;
  kvBlockAllocator allocator(cfg.total_kv_blocks);

  if (cfg.prefill_chunk_size > cfg.max_batch_tokens) {
    std::cerr << "Error: prefill_chunk_size must be less than or equal to max_batch_tokens\n";
    std::exit(1);
  }

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

    for (int i = (int)active.size() - 1; i >= 0; --i) {
      if (active[i].finished) {
        allocator.release(active[i].kv_blocks_used);
        active.erase(active.begin() + i);
      } else if (active[i].preempted) {
        allocator.release(active[i].kv_blocks_used);
        active[i].kv_blocks_used = 0;
        active[i].prefill_done = false;
        active[i].prompt_tokens_processed = 0;
        active[i].tokens_generated = 0;
        active[i].preempted = false;
        waiting.push_back(std::move(active[i].request));
        active.erase(active.begin() + i);
      }
    }
     
    int budget_used = 0;
    for (const auto& s : active) {
      budget_used += slot_cost(s, cfg.prefill_chunk_size);
    }
    int budget_remaining = cfg.max_batch_tokens - budget_used;
    int free_slots = cfg.max_slots - (int)active.size();
    bool was_idle = active.empty();
    
    // admit new requests
    if (free_slots > 0 && budget_remaining > 0 && !waiting.empty()) {
      std::vector<int> picks = policy.pick(waiting, free_slots, budget_remaining, cfg.prefill_chunk_size, t);
      if (was_idle && !picks.empty()) {
        double max_arrival = 0.0;
        for (int idx : picks) {
          max_arrival = std::max(max_arrival, waiting[idx].arrival_ms);
        }
        t = std::max(t, max_arrival);
      }
      std::vector<int> admitted_picks;
      int available_blocks=allocator.free_blocks();
      for (int idx: picks) {
        int prompt_blocks = (waiting[idx].prompt_length+cfg.kv_block_size-1)/cfg.kv_block_size;
        if (available_blocks >= prompt_blocks) {
          admitted_picks.push_back(idx);
          available_blocks-=prompt_blocks;
        }
      }
      std::sort(admitted_picks.begin(), admitted_picks.end(), [](int a, int b) { return a > b; });
      for (int idx: admitted_picks) {
        Slot s;
        s.request = std::move(waiting[idx]);
        s.true_out_len = sample_out_length(rng);
        s.start_ms = t;
        s.prompt_tokens_processed = 0;
        s.prefill_done = false;
        s.tokens_generated = 0;
        //allocate initial kv blocks
        int prompt_blocks=(s.request.prompt_length+cfg.kv_block_size-1)/cfg.kv_block_size;
        s.kv_blocks_used= allocator.allocate(prompt_blocks);

        if(run_start<0) run_start=t;
        active.push_back(std::move(s));
        waiting.erase(waiting.begin()+idx);

      }
    }

    if (active.empty()) continue;

    // calculate tokens processed in this tick
    int tokens_tick = 0;
    for (const auto& s : active) {
      tokens_tick += slot_cost(s, cfg.prefill_chunk_size);
    }

    // step time: total tokens processed
    double step_ms = cfg.d0 + cfg.d1 * tokens_tick;
    t += step_ms;
    total_ticks++;
    occupancy.push_back((int)active.size());

    for (auto& s : active) {
      if (s.finished || s.preempted) continue;
      if (!s.prefill_done) {
        int remaining_prompt = s.request.prompt_length - s.prompt_tokens_processed;
        int chunk = std::min(cfg.prefill_chunk_size, remaining_prompt);
        s.prompt_tokens_processed += chunk;
        tokens_made += chunk;
        if (s.prompt_tokens_processed >= s.request.prompt_length) {
          s.prefill_done = true;
          s.first_token_ms = t;
          s.tokens_generated = 1; // first free token
        }
      } else {

        // DECODE
        s.tokens_generated++;
        tokens_made++;
        int total_tokens=s.request.prompt_length+ s.tokens_generated;
        int blocks_needed=(total_tokens+cfg.kv_block_size-1)/cfg.kv_block_size;
        if(blocks_needed>s.kv_blocks_used){
          int extra_blocks=blocks_needed-s.kv_blocks_used;
          int got=allocator.allocate(extra_blocks);
          s.kv_blocks_used+=got;
          int still_need = extra_blocks - got;
          if (still_need > 0) {
            // KV Preemption
            for (auto& victim : active) {
              if (!victim.finished && !victim.preempted && victim.request.tier > s.request.tier) {
                victim.preempted = true;
                stats.total_preemptions++;
                allocator.release(victim.kv_blocks_used);
                victim.kv_blocks_used = 0;
                int grab = std::min(still_need, allocator.free_blocks());
                allocator.allocate(grab);
                s.kv_blocks_used += grab;
                still_need -= grab;
                if (still_need == 0) break;
              }
            }
            if (still_need > 0) {
              s.preempted = true;
              stats.total_preemptions++;
              allocator.release(s.kv_blocks_used);
              s.kv_blocks_used = 0;
              s.tokens_generated--; 
              tokens_made--;
              continue;
            }
          }
        }
        if (s.tokens_generated>= s.true_out_len) {
          s.finished = true;
          s.end_ms = t;
          s.request.promise.set_value(Result{
              s.request.id, s.request.tier, s.request.prompt_length, s.true_out_len,
              s.request.arrival_ms, s.start_ms, s.first_token_ms, s.end_ms});
        }
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

  stats.total_ticks = total_ticks;
  stats.tokens_made = tokens_made;
  stats.avg_occupancy = avg_occupancy;
  stats.duration_ms = duration;
  stats.throughput_tok_s = throughput;
  stats.idle_seat_ticks = idle_ticks;
  stats.total_seat_ticks = (long)cfg.max_slots * total_ticks;
  stats.utilization = cfg.max_slots > 0 ? avg_occupancy / cfg.max_slots : 0.0;

  std::cerr << "[continuous] ticks=" << total_ticks << " tokens=" << tokens_made
            << " duration_ms=" << duration << " throughput_tok_s=" << throughput
            << " avg_occupancy=" << avg_occupancy << "/" << cfg.max_slots
            << " idle_seat_ticks=" << idle_ticks
            << " preemptions=" << stats.total_preemptions << "\n";
}


double get_tier_slo(int tier) {
  if (tier == 0) return 200.0;   
  if (tier == 1) return 600.0;  
  return 2000.0;                  
}

struct TraceEntry {
  double arrival_ms; 
  int prompt_length;
  int tier;          // 0=Enterprise,1 = Pro, 2 = Free
};

std::vector<TraceEntry> generate_trace(unsigned seed, int n, double lambda) {
  std::mt19937 rng(seed);
  std::exponential_distribution<> gap_s(lambda);
  std::uniform_int_distribution<> prompt_len(10, 500);
  std::discrete_distribution<> tier_dist({0.2, 0.3, 0.5}); // 20% Enterprise, 30% Pro, 50% Free
  std::vector<TraceEntry> trace;
  trace.reserve(n);
  double t = 0.0;
  for (int i = 0; i < n; ++i) {
    t += gap_s(rng) * 1000.0;
    trace.push_back({ t, prompt_len(rng), tier_dist(rng) });
  }
  return trace;
}

// stats
double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  double rank = (p / 100.0) * (double)(values.size() - 1);
  size_t lo = (size_t)std::floor(rank);
  size_t hi = (size_t)std::ceil(rank);
  if (lo == hi) return values[lo];
  double frac = rank - (double)lo;
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

struct LatencySummary {
  double p50 = 0, p95 = 0, p99 = 0, mean = 0;
};

LatencySummary summarize(const std::vector<double>& values) {
  LatencySummary s;
  s.p50 = percentile(values, 50);
  s.p95 = percentile(values, 95);
  s.p99 = percentile(values, 99);
  s.mean = values.empty() ? 0.0
      : std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  return s;
}

bool file_is_empty(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return true; // doesn't exist -> "empty"
  return st.st_size == 0;
}


struct CliOptions {
  std::string mode = "continuous";
  std::string policy = "fcfs";
  double lambda = 5.0;
  unsigned seed = 42;
  std::string out = "results.csv";      
  std::string summary = "summary.csv";   
  int n = 200;
  int max_batch_tokens = 512;
  int prefill_chunk_size = 256;
  bool pace = true;
  bool verbose = false;
  bool help = false;
  int kv_block_size=16;
  int total_kv_blocks=256;
  double ttft_slo_ms = 500.0;

};

void print_usage() {
  std::cerr <<
    "usage: ./main [--mode=continuous|static] [--policy=fcfs|shortprompt]\n"
    "              [--lambda=5.0] [--seed=42] [--n=200] [--pace=1]\n"
    "              [--max_batch_tokens=512] [--chunk_size=256]\n"
    "              [--out=results.csv|none] [--summary=summary.csv]\n"
    "              [--verbose]\n"
    "\n"
    "  --out=none skips writing per-request rows (useful for large\n"
    "             multi-seed sweeps where you only need the summary row).\n"
    "  --summary  is APPENDED to, one row per run, so it can be pointed\n"
    "             at the same file across many seeds/policies/lambdas.\n";
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { opt.help = true; continue; }
    if (arg.rfind("--", 0) != 0) {
      std::cerr << "warning: ignoring unrecognized argument '" << arg << "'\n";
      continue;
    }
    std::string body = arg.substr(2);
    auto eq = body.find('=');
    std::string key, val;
    if (eq == std::string::npos) {
      key = body;
      if (key != "verbose" && i + 1 < argc) val = argv[++i];
    } else {
      key = body.substr(0, eq);
      val = body.substr(eq + 1);
    }

    if (key == "mode") opt.mode = val;
    else if (key == "policy") opt.policy = val;
    else if (key == "lambda") opt.lambda = std::stod(val);
    else if (key == "seed") opt.seed = (unsigned)std::stoul(val);
    else if (key == "out") opt.out = val;
    else if (key == "summary") opt.summary = val;
    else if (key == "n") opt.n = std::stoi(val);
    else if (key == "max_batch_tokens") opt.max_batch_tokens = std::stoi(val);
    else if (key == "chunk_size") opt.prefill_chunk_size = std::stoi(val);
    else if (key == "pace") opt.pace = (val == "1" || val == "true");
    else if (key == "verbose") opt.verbose = val.empty() || val == "1" || val == "true";
    else if (key == "kv_block_size") opt.kv_block_size = std::stoi(val);
    else if (key == "total_kv_blocks") opt.total_kv_blocks = std::stoi(val);
    else if (key == "ttft_slo") opt.ttft_slo_ms = std::stod(val);
    else std::cerr << "warning: unknown flag --" << key << "\n";
  }
  return opt;
}

int main(int argc, char** argv) {
  program_start = std::chrono::steady_clock::now();
  CliOptions opt = parse_args(argc, argv);
  if (opt.help) { print_usage(); return 0; }

  EngineConfig config;
  if (opt.mode == "continuous") config.continuous = true;
  else if (opt.mode == "static") config.continuous = false;
  else {
    std::cerr << "warning: unknown --mode '" << opt.mode << "', using continuous\n";
    config.continuous = true;
  }
  config.seed = opt.seed;
  config.pace_with_sleeps = opt.pace;
  config.max_batch_tokens = opt.max_batch_tokens;
  config.prefill_chunk_size = opt.prefill_chunk_size;
  config.kv_block_size = opt.kv_block_size;
  config.total_kv_blocks = opt.total_kv_blocks;
  config.ttft_slo_ms = opt.ttft_slo_ms;

    std::unique_ptr<AdmissionPolicy> policy;
  if (opt.policy == "shortprompt") {
    policy = std::make_unique<ShortPromptFirstPolicy>();
  } else if (opt.policy == "slo") {
    policy = std::make_unique<SLOPolicy>();
  } else {
    if (opt.policy != "fcfs")
      std::cerr << "warning: unknown --policy '" << opt.policy << "', using fcfs\n";
    policy = std::make_unique<FCFSPolicy>();
  }


  ThreadSafeQueue<Request> queue;
  RunStats stats;
  std::thread worker_thread(config.continuous ? worker_continuous : worker_static,
                             std::ref(queue), std::ref(config), std::ref(*policy),
                             std::ref(stats));

  auto trace = generate_trace(opt.seed, opt.n, opt.lambda);
  std::vector<std::future<Result>> futures;
  futures.reserve(opt.n);

  for (int i = 0; i < opt.n; ++i) {
    Request req;
    req.id = i;
    req.tier = trace[i].tier;
    req.prompt_length = trace[i].prompt_length;
    req.max_tokens = config.max_tokens;
    req.arrival_ms = trace[i].arrival_ms;
    req.ttft_deadline_ms = req.arrival_ms + get_tier_slo(req.tier);

    if (opt.pace) {
      double now = now_ms();
      if (trace[i].arrival_ms > now)
        std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(trace[i].arrival_ms - now));
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

  // per request output
  if (opt.out != "none" && !opt.out.empty()) {
    std::ofstream out(opt.out);
    if (!out) {
      std::cerr << "error: could not open '" << opt.out << "' for writing\n";
      return 1;
    }
    out << std::fixed << std::setprecision(3);
    out << "mode,policy,lambda,seed,id,tier,prompt_length,true_out_len,"
           "arrival_ms,start_ms,first_token_ms,end_ms\n";
    for (auto& r : results) {
      out << opt.mode << ',' << opt.policy << ',' << opt.lambda << ',' << opt.seed << ','
          << r.id << ',' << r.tier << ',' << r.prompt_length << ',' << r.true_out_len << ','
          << r.arrival_ms << ',' << r.start_ms << ',' << r.first_token_ms << ',' << r.end_ms
          << '\n';
      if (opt.verbose) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "req " << r.id << " (tier " << r.tier << ")"
                   << " | wait " << r.start_ms - r.arrival_ms
                   << " ms | TTFT " << r.first_token_ms - r.arrival_ms
                   << " ms | total " << r.end_ms - r.arrival_ms << " ms\n";
      }
    }
    std::cerr << "wrote " << results.size() << " rows to " << opt.out << "\n";
  }

  std::vector<double> waits, ttfts, latencies;
  waits.reserve(results.size());
  ttfts.reserve(results.size());
  latencies.reserve(results.size());
  for (auto& r : results) {
    waits.push_back(r.start_ms - r.arrival_ms);
    ttfts.push_back(r.first_token_ms - r.arrival_ms);
    latencies.push_back(r.end_ms - r.arrival_ms);
  }
  LatencySummary wait_s = summarize(waits);
  LatencySummary ttft_s = summarize(ttfts);
  LatencySummary lat_s = summarize(latencies);

  std::cerr << std::fixed << std::setprecision(3)
            << "[summary] wait p50/p95/p99 = " << wait_s.p50 << "/" << wait_s.p95 << "/" << wait_s.p99
            << " ms | TTFT p50/p95/p99 = " << ttft_s.p50 << "/" << ttft_s.p95 << "/" << ttft_s.p99
            << " ms | latency p50/p95/p99 = " << lat_s.p50 << "/" << lat_s.p95 << "/" << lat_s.p99 << " ms\n";

  if (opt.summary != "none" && !opt.summary.empty()) {
    bool need_header = file_is_empty(opt.summary);
    std::ofstream sum(opt.summary, std::ios::app);
    if (!sum) {
      std::cerr << "error: could not open '" << opt.summary << "' for writing\n";
      return 1;
    }
    if (need_header) {
      sum << "mode,policy,lambda,seed,n,throughput_tok_s,avg_occupancy,utilization,"
             "wait_p50_ms,wait_p95_ms,wait_p99_ms,wait_mean_ms,"
             "ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,ttft_mean_ms,"
             "latency_p50_ms,latency_p95_ms,latency_p99_ms,latency_mean_ms\n";
    }
    sum << std::fixed << std::setprecision(4);
    sum << opt.mode << ',' << opt.policy << ',' << opt.lambda << ',' << opt.seed << ','
        << results.size() << ',' << stats.throughput_tok_s << ',' << stats.avg_occupancy << ','
        << stats.utilization << ','
        << wait_s.p50 << ',' << wait_s.p95 << ',' << wait_s.p99 << ',' << wait_s.mean << ','
        << ttft_s.p50 << ',' << ttft_s.p95 << ',' << ttft_s.p99 << ',' << ttft_s.mean << ','
        << lat_s.p50 << ',' << lat_s.p95 << ',' << lat_s.p99 << ',' << lat_s.mean << '\n';
    std::cerr << "appended summary row to " << opt.summary << "\n";
  }

  int tier_violations[3] = {0, 0, 0};
  int tier_counts[3] = {0, 0, 0};
  for (auto& r : results) {
    tier_counts[r.tier]++;
    double ttft = r.first_token_ms - r.arrival_ms;
    if (ttft > get_tier_slo(r.tier)) tier_violations[r.tier]++;
  }

  const char* tier_names[] = {"Enterprise (200ms)", "Pro (600ms)", "Free (2000ms)"};
  int total_violations = 0;
  for (int t = 0; t < 3; ++t) {
    total_violations += tier_violations[t];
    if (tier_counts[t] == 0) continue;
    double att = 100.0 * (1.0 - (double)tier_violations[t] / tier_counts[t]);
    std::cerr << "[slo] " << tier_names[t] << ": attainment=" << std::fixed << std::setprecision(1)
              << att << "% (" << (tier_counts[t] - tier_violations[t]) << "/" << tier_counts[t] << " met)\n";
  }
  double overall_att = 100.0 * (1.0 - (double)total_violations / results.size());
  std::cerr << "[slo] Overall: attainment=" << std::fixed << std::setprecision(1)
            << overall_att << "% (" << (results.size() - total_violations) << "/" << results.size() << " met)\n";

  return 0;
}