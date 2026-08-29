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
#include <unordered_map>
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
  int shared_len;
};

struct Request {
  int id;
  int tier = 0;
  int prompt_length;
  int max_tokens;
  double arrival_ms;
  std::promise<Result> promise;
  double ttft_deadline_ms = 0.0;
  int template_id;
  int shared_len;
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
  int num_templates = 5;
  bool prefix_cache = false;
  std::string eviction = "lru";
  bool quadratic_cost = false;
};

//aggregate stats by a worker for 1 run
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
  int cached_prefix_blocks = 0;
  bool needs_cache_insert = false;
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

struct PrefixCacheEntry {
    int template_id;
    int blocks_used;       
    int ref_count = 0;      
    int last_used_tick = 0; 
    int shared_len = 0;     
    double gds_priority = 0.0; 
};

struct PrefixCache {
    std::unordered_map<int, PrefixCacheEntry> entries;
    long hits = 0;
    long misses = 0;

    PrefixCacheEntry* lookup(int template_id) {
        auto it = entries.find(template_id);
        if (it != entries.end()) return &it->second;
        return nullptr;
    }

    const PrefixCacheEntry* lookup(int template_id) const {
        auto it = entries.find(template_id);
        if (it != entries.end()) return &it->second;
        return nullptr;
    }

    double hit_rate() const {
        long total = hits + misses;
        return total > 0 ? (double)hits / total : 0.0;
    }
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
  return 1; //1 decode token
}

// effective first chunk cost of admitting each waiting request, in
// batch token budget units. If request template already has a cached
// prefix, only its non shared/unique tokens need to be prefilled

std::vector<int> compute_first_chunk_costs(const std::vector<Request>& waiting,
                                            int chunk_size,
                                            const PrefixCache* prefix_cache) {
  std::vector<int> costs(waiting.size());
  for (size_t i = 0; i < waiting.size(); i++) {
    int effective_prompt_length = waiting[i].prompt_length;
    if (prefix_cache && waiting[i].template_id >= 0) {
      const PrefixCacheEntry* cached = prefix_cache->lookup(waiting[i].template_id);
      if (cached && cached->blocks_used > 0) {
        effective_prompt_length = std::max(0, waiting[i].prompt_length - waiting[i].shared_len);
      }
    }
    costs[i] = std::min(effective_prompt_length, chunk_size);
  }
  return costs;
}

struct AdmissionPolicy {
  virtual std::vector<int> pick(
      const std::vector<Request>& waiting,
      const std::vector<int>& first_chunk_costs,
      int free_slots,
      int budget_remaining,
      double current_time) = 0;
  virtual ~AdmissionPolicy() = default;
};

struct FCFSPolicy : AdmissionPolicy {
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      const std::vector<int>& first_chunk_costs,
      int free_slots,
      int budget_remaining,
      double /*current_time*/) override {
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i = 0; i < (int)waiting.size() && (int)idx.size() < free_slots; i++) {
      int cost = first_chunk_costs[i];
      if (cost <= budget) {
        idx.push_back(i);
        budget -= cost;
      }
    }
    return idx;
  }
};

struct ShortPromptFirstPolicy : AdmissionPolicy {
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      const std::vector<int>& first_chunk_costs,
      int free_slots,
      int budget_remaining,
      double /*current_time*/) override {
    std::vector<int> order;
    for (int i = 0; i < (int)waiting.size(); i++) order.push_back(i);
    //sorting by effective post cache cost instead of just raw prompt length
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return first_chunk_costs[a] < first_chunk_costs[b];
    });
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i : order) {
      int cost = first_chunk_costs[i];
      if (cost > budget) break;
      idx.push_back(i);
      budget -= cost;
      if ((int)idx.size() >= free_slots) break;
    }
    std::sort(idx.begin(), idx.end());
    return idx;
  }
};

struct SLOPolicy : AdmissionPolicy{
  std::vector<int> pick(
      const std::vector<Request>& waiting,
      const std::vector<int>& first_chunk_costs,
      int free_slots,
      int budget_remaining,
      double current_time) override {
   
    std::vector<int> order;
    for (int i = 0; i < (int)waiting.size(); i++) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      double slack_a = waiting[a].ttft_deadline_ms - current_time;
      double slack_b = waiting[b].ttft_deadline_ms - current_time;
      return slack_a < slack_b;  //smaller slack= more urgent = goes first
    });
 
    std::vector<int> idx;
    int budget = budget_remaining;
    for (int i : order) {
      if ((int)idx.size() >= free_slots) break;
      int cost = first_chunk_costs[i];
      if (cost <= budget) {
        idx.push_back(i);
        budget -= cost;
      }
    }
    std::sort(idx.begin(), idx.end()); 
    return idx;
  }

};

struct PromptTemplate {
    int id;
    int shared_len;  
};

PromptTemplate templates[] = {
    {0, 30},  
    {1, 80},   
    {2, 200},  
    {3, 400},
    {4, 50},   
};


void worker_static(ThreadSafeQueue<Request>& q, const EngineConfig& config,
                    AdmissionPolicy& policy, RunStats& stats) {
  std::mt19937 rng(config.seed);
  std::vector<Request> waiting;
  bool done = false;
  long total_idle = 0, total_seat_ticks = 0;
  int total_batches = 0;
  double sim_clock = 0.0;

  while (true) {
    while (!done) {
      if (waiting.empty()) {
        Request first = q.pop();
        if (first.id == -1) { done = true; break; }
        waiting.push_back(std::move(first));
      } else {
        auto m = q.try_pop_for(std::chrono::milliseconds(0));
        if (!m) break;
        if (m->id == -1) { done = true; break; }
        waiting.push_back(std::move(*m));
      }
    }

    if (waiting.empty() && done) break;

    double min_arr = 1e18;
    for (const auto& r : waiting) min_arr = std::min(min_arr, r.arrival_ms);
    if (sim_clock < min_arr) sim_clock = min_arr;

    std::vector<Request> arrived;
    std::vector<Request> future_reqs;
    for (auto& r : waiting) {
      if (r.arrival_ms <= sim_clock + (config.pace_with_sleeps ? config.batch_timeout_ms : 0.0)) {
        arrived.push_back(std::move(r));
      } else {
        future_reqs.push_back(std::move(r));
      }
    }
    waiting = std::move(future_reqs);

    if (arrived.empty()) {
      if (!waiting.empty()) {
        sim_clock = min_arr;
        continue;
      }
      if (done) break;
      continue;
    }

    std::vector<int> first_chunk_costs = compute_first_chunk_costs(arrived, config.prefill_chunk_size, nullptr);
    std::vector<int> picks = policy.pick(arrived, first_chunk_costs, config.max_slots, config.max_batch_tokens, sim_clock);
    std::sort(picks.begin(), picks.end(), [](int a, int b) { return a > b; });

    std::vector<Slot> batch;
    for (int idx : picks) {
      Slot s;
      s.request = std::move(arrived[idx]);
      batch.push_back(std::move(s));
      arrived.erase(arrived.begin() + idx); 
    }
    for (auto& r : arrived) waiting.push_back(std::move(r));

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
              s.start_ms, s.first_token_ms, s.end_ms, s.request.shared_len});
        }
      }
    }

    long idle = 0;
    for (auto& s : batch) idle += ticks - s.finish_tick;
    total_idle += idle;
    total_seat_ticks += (long)seats * ticks;
    ++total_batches;
    sim_clock = t;

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

double compute_cost(int shared_len, const EngineConfig& cfg) {
    double cost = cfg.p0 + cfg.p1 * shared_len;
    if (cfg.quadratic_cost) {
        cost += 0.001 * shared_len * shared_len;
    }
    return cost;
}

void evict_for_blocks(int needed, kvBlockAllocator& allocator, PrefixCache& prefix_cache, const EngineConfig& cfg, int current_tick, double& gds_L) {
    while (allocator.free_blocks() < needed) {
        auto best_it = prefix_cache.entries.end();
        double best_score = -1e9;
        
        for (auto it = prefix_cache.entries.begin(); it != prefix_cache.entries.end(); ++it) {
            if (it->second.ref_count > 0) continue;
            
            double score = 0;
            if (cfg.eviction == "lru") {
                score = current_tick - it->second.last_used_tick; 
            } else if (cfg.eviction == "cost_ratio") {
                score = (double)(current_tick - it->second.last_used_tick) / compute_cost(it->second.shared_len, cfg);
            } else if (cfg.eviction == "gds") {
                score = -it->second.gds_priority;
            }
            
            if (score > best_score) {
                best_score = score;
                best_it = it;
            }
        }
        
        if (best_it == prefix_cache.entries.end()) break; 
        
        if (cfg.eviction == "gds") {
            gds_L = best_it->second.gds_priority;
        }
        
        allocator.release(best_it->second.blocks_used);
        prefix_cache.entries.erase(best_it);
    }
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
  PrefixCache prefix_cache;
  double gds_L = 0.0;

  if (cfg.prefill_chunk_size > cfg.max_batch_tokens) {
    std::cerr << "Error: prefill_chunk_size must be less than or equal to max_batch_tokens\n";
    std::exit(1);
  }

  while (!(done && waiting.empty() && active.empty())) {
    while (!done) {
      if (active.empty() && waiting.empty()) {
        Request first = q.pop();
        if (first.id == -1) { done = true; break; }
        waiting.push_back(std::move(first));
      } else {
        auto m = q.try_pop_for(std::chrono::milliseconds(0));
        if (!m) break;
        if (m->id == -1) { done = true; break; }
        waiting.push_back(std::move(*m));
      }
    }

    for (int i = (int)active.size() - 1; i >= 0; --i) {
      if (active[i].finished) {
        allocator.release(active[i].kv_blocks_used);
        if (cfg.prefix_cache && active[i].cached_prefix_blocks > 0) {
            auto* cached = prefix_cache.lookup(active[i].request.template_id);
            if (cached) cached->ref_count--;
        }
        active.erase(active.begin() + i);
      } else if (active[i].preempted) {
        allocator.release(active[i].kv_blocks_used);
        active[i].kv_blocks_used = 0;
        active[i].cached_prefix_blocks = 0;
        active[i].needs_cache_insert = false;
        active[i].prefill_done = false;
        active[i].prompt_tokens_processed = 0;
        active[i].tokens_generated = 0;
        active[i].preempted = false;
        waiting.push_back(std::move(active[i].request));
        active.erase(active.begin() + i);
      }
    }

    if (done && waiting.empty() && active.empty()) break;

    if (active.empty() && !waiting.empty()) {
      double min_arr = 1e18;
      for (const auto& r : waiting) min_arr = std::min(min_arr, r.arrival_ms);
      if (t < min_arr) t = min_arr;
    }

    std::vector<Request> arrived;
    std::vector<Request> future_reqs;
    for (auto& r : waiting) {
      if (r.arrival_ms <= t) {
        arrived.push_back(std::move(r));
      } else {
        future_reqs.push_back(std::move(r));
      }
    }
    waiting = std::move(future_reqs);

    int budget_used = 0;
    for (const auto& s : active) {
      budget_used += slot_cost(s, cfg.prefill_chunk_size);
    }
    int budget_remaining = cfg.max_batch_tokens - budget_used;
    int free_slots = cfg.max_slots - (int)active.size();

    // admit new requests
    if (free_slots > 0 && budget_remaining > 0 && !arrived.empty()) {
      std::vector<int> first_chunk_costs = compute_first_chunk_costs(
          arrived, cfg.prefill_chunk_size, cfg.prefix_cache ? &prefix_cache : nullptr);
      std::vector<int> picks = policy.pick(arrived, first_chunk_costs, free_slots, budget_remaining, t);
      
      std::vector<int> admitted_picks;
      std::unordered_map<int, int> reserved_blocks;
      std::unordered_map<int, PrefixCacheEntry*> pending_hits;
      int decode_headroom = (int)active.size();

      for (int idx: picks) {
        int required_blocks = (arrived[idx].prompt_length+cfg.kv_block_size-1)/cfg.kv_block_size;
        auto* cached = (cfg.prefix_cache && arrived[idx].template_id >= 0) ? prefix_cache.lookup(arrived[idx].template_id) : nullptr;
        bool tentative_hit = false;
        if (cached && cached->blocks_used > 0) {
            int unique_len = arrived[idx].prompt_length - arrived[idx].shared_len;
            required_blocks = (unique_len + cfg.kv_block_size - 1) / cfg.kv_block_size;
            cached->ref_count++;
            tentative_hit = true;
        }

        if (allocator.free_blocks() < required_blocks + decode_headroom && cfg.prefix_cache) {
            evict_for_blocks(required_blocks + decode_headroom, allocator, prefix_cache, cfg, total_ticks, gds_L);
        }

        if (allocator.free_blocks() >= required_blocks + decode_headroom || (active.empty() && allocator.free_blocks() >= required_blocks)) {
          admitted_picks.push_back(idx);
          allocator.allocate(required_blocks);
          reserved_blocks[idx] = required_blocks;
          if (tentative_hit) pending_hits[idx] = cached;
        } else if (tentative_hit) {
          cached->ref_count--;
        }
      }
      
      std::sort(admitted_picks.begin(), admitted_picks.end(), [](int a, int b) { return a > b; });
      for (int idx: admitted_picks) {
        Slot s;
        s.request = std::move(arrived[idx]);
        s.true_out_len = sample_out_length(rng);
        s.start_ms = t;
        s.prompt_tokens_processed = 0;
        s.prefill_done = false;
        s.tokens_generated = 0;
        s.kv_blocks_used = reserved_blocks[idx];
        s.cached_prefix_blocks = 0;
        s.needs_cache_insert = false;
        
        auto hit_it = pending_hits.find(idx);
        if (hit_it != pending_hits.end()) {
            PrefixCacheEntry* cached = hit_it->second;
            prefix_cache.hits++;
            cached->last_used_tick = total_ticks;
            s.prompt_tokens_processed = cached->shared_len;
            s.cached_prefix_blocks = cached->blocks_used;
        } else {
            prefix_cache.misses++;
            if (cfg.prefix_cache && s.request.template_id >= 0) {
                s.needs_cache_insert = true;
            }
        }

        if (run_start < 0) run_start = t;
        active.push_back(std::move(s));
        arrived.erase(arrived.begin() + idx);
      }
    }

    for (auto& r : arrived) {
      waiting.push_back(std::move(r));
    }

    if (active.empty()) continue;

    int tokens_tick = 0;
    for (const auto& s : active) {
      tokens_tick += slot_cost(s, cfg.prefill_chunk_size);
    }

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
          s.tokens_generated = 1;
          
          if (cfg.prefix_cache && s.needs_cache_insert && s.request.template_id >= 0) {
              int shared_blocks = (s.request.shared_len + cfg.kv_block_size - 1) / cfg.kv_block_size;
              auto* existing = prefix_cache.lookup(s.request.template_id);
              if (existing) {
                  existing->ref_count++;
                  existing->last_used_tick = total_ticks;
                  int give_back = std::min(shared_blocks, s.kv_blocks_used);
                  allocator.release(give_back);
                  s.kv_blocks_used -= give_back;
                  s.cached_prefix_blocks = existing->blocks_used;
              } else {
                  PrefixCacheEntry entry;
                  entry.template_id = s.request.template_id;
                  entry.blocks_used = shared_blocks;
                  entry.ref_count = 1;
                  entry.shared_len = s.request.shared_len;
                  entry.last_used_tick = total_ticks;
                  entry.gds_priority = shared_blocks > 0
                      ? compute_cost(entry.shared_len, cfg) / entry.blocks_used + gds_L
                      : gds_L;
                  prefix_cache.entries[entry.template_id] = entry;

                  s.cached_prefix_blocks = shared_blocks;
                  s.kv_blocks_used -= shared_blocks;
              }
              s.needs_cache_insert = false;
          }
        }
      } else {
        // DECODE
        s.tokens_generated++;
        tokens_made++;
        int total_tokens=s.request.prompt_length+ s.tokens_generated;
        int blocks_needed = (total_tokens + cfg.kv_block_size - 1) / cfg.kv_block_size;
        int blocks_held = s.kv_blocks_used + s.cached_prefix_blocks;
        
        if (blocks_needed > blocks_held) {
          int extra_blocks = blocks_needed - blocks_held;
          
          if (cfg.prefix_cache && allocator.free_blocks() < extra_blocks) {
              evict_for_blocks(extra_blocks, allocator, prefix_cache, cfg, total_ticks, gds_L);
          }
          
          int got = allocator.allocate(extra_blocks);
          s.kv_blocks_used += got;
          int still_need = extra_blocks - got;
          if (still_need > 0) {
            while (still_need > 0) {
              auto victim_it = active.end();
              for (auto it = active.begin(); it != active.end(); ++it) {
                if (it->finished || it->preempted || it->request.tier <= s.request.tier) continue;
                if (victim_it == active.end() || it->request.tier > victim_it->request.tier) {
                  victim_it = it;
                }
              }
              if (victim_it == active.end()) break;

              victim_it->preempted = true;
              stats.total_preemptions++;
              allocator.release(victim_it->kv_blocks_used);
              victim_it->kv_blocks_used = 0;

              if (cfg.prefix_cache && victim_it->cached_prefix_blocks > 0) {
                  auto* cached = prefix_cache.lookup(victim_it->request.template_id);
                  if (cached) cached->ref_count--;
                  victim_it->cached_prefix_blocks = 0;
              }

              if (cfg.prefix_cache && allocator.free_blocks() < still_need) {
                  evict_for_blocks(still_need, allocator, prefix_cache, cfg, total_ticks, gds_L);
              }

              int grab = std::min(still_need, allocator.free_blocks());
              allocator.allocate(grab);
              s.kv_blocks_used += grab;
              still_need -= grab;
            }
            if (still_need > 0) {
              s.preempted = true;
              stats.total_preemptions++;
              allocator.release(s.kv_blocks_used);
              s.kv_blocks_used = 0;
              if (cfg.prefix_cache && s.cached_prefix_blocks > 0) {
                  auto* cached = prefix_cache.lookup(s.request.template_id);
                  if (cached) cached->ref_count--;
                  s.cached_prefix_blocks = 0;
              }
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
              s.request.arrival_ms, s.start_ms, s.first_token_ms, s.end_ms, s.request.shared_len});
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
            
  if (cfg.prefix_cache) {
      std::cerr << "[prefix_cache] hits=" << prefix_cache.hits 
                << " misses=" << prefix_cache.misses 
                << " hit_rate=" << (prefix_cache.hit_rate() * 100.0) << "%\n";
  }
}


double get_tier_slo(int tier) {
  if (tier == 0) return 200.0;   
  if (tier == 1) return 600.0;  
  return 2000.0;                  
}

struct TraceEntry {
  double arrival_ms; 
  int prompt_length; 
  int tier; // 0=Enterprise,1 = Pro, 2 = Free
  int template_id;
  int shared_len; 
};

std::vector<TraceEntry> generate_trace(unsigned seed, int n, double lambda, int num_templates) {
  std::mt19937 rng(seed);
  std::exponential_distribution<> gap_s(lambda);
  std::uniform_int_distribution<> unique_len_dist(10, 150);
  std::discrete_distribution<> tier_dist({0.2, 0.3, 0.5}); //20% Enterprise, 30% Pro, 50% Free
  std::vector<TraceEntry> trace;
  trace.reserve(n);
  double t = 0.0;
  std::uniform_int_distribution<> template_dist(0, std::max(0, num_templates - 1));

  for (int i = 0; i < n; ++i) {
    t += gap_s(rng) * 1000.0;
    int template_id = -1;
    int shared_len = 0;
    int unique_len = unique_len_dist(rng);
    if (num_templates > 0) {
      template_id = template_dist(rng);
      shared_len = templates[template_id % 5].shared_len; //5 templates defined
    } else {
      unique_len = std::uniform_int_distribution<>(10, 500)(rng);
    }
    int prompt_length = shared_len + unique_len;
    trace.push_back({ t, prompt_length, tier_dist(rng), template_id, shared_len });
  }
  return trace;
}

//stats
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
  if (stat(path.c_str(), &st) != 0) return true; 
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
  int num_templates = 5;
  bool prefix_cache = false;
  std::string eviction = "lru";
  bool quadratic_cost = false;
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
    else if (key == "num_templates") opt.num_templates = std::stoi(val);
    else if (key == "prefix_cache") opt.prefix_cache = (val == "1" || val == "true");
    else if (key == "eviction") opt.eviction = val;
    else if (key == "quadratic_cost") opt.quadratic_cost = (val == "1" || val == "true");
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
  config.num_templates = opt.num_templates;
  config.prefix_cache = opt.prefix_cache;
  config.eviction = opt.eviction;
  config.quadratic_cost = opt.quadratic_cost;

  if (!config.continuous && config.prefix_cache) {
    std::cerr << "warning: --mode=static ignores --prefix_cache/--kv_block_size/"
                 "--total_kv_blocks/--eviction (static mode has no KV manager); "
                 "these flags will have no effect.\n";
  }

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

  auto trace = generate_trace(opt.seed, opt.n, opt.lambda, opt.num_templates);

  if (config.continuous) {
    for (const auto& te : trace) {
      int req_blocks = (te.prompt_length + config.kv_block_size - 1) / config.kv_block_size;
      if (req_blocks > config.total_kv_blocks) {
        std::cerr << "error: a generated request needs " << req_blocks
                  << " KV blocks (prompt_length=" << te.prompt_length
                  << ") but total_kv_blocks=" << config.total_kv_blocks
                  << ". It could never be admitted and the run would hang forever. "
                  << "Increase --total_kv_blocks or --kv_block_size.\n";
        return 1;
      }
    }
  }

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
    req.template_id = trace[i].template_id;
    req.shared_len = trace[i].shared_len;

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
           "arrival_ms,start_ms,first_token_ms,end_ms,shared_len\n";
    for (auto& r : results) {
      out << opt.mode << ',' << opt.policy << ',' << opt.lambda << ',' << opt.seed << ','
          << r.id << ',' << r.tier << ',' << r.prompt_length << ',' << r.true_out_len << ','
          << r.arrival_ms << ',' << r.start_ms << ',' << r.first_token_ms << ',' << r.end_ms << ','
          << r.shared_len
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