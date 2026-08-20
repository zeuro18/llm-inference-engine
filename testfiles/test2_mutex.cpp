#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>
int shared_count=0;
const int INCREMENTS_THREAD=100000;
std::mutex mtx;
void worker(){
    
    for(int i=0;i<INCREMENTS_THREAD;i++){
        std::lock_guard<std::mutex> lock(mtx);
        shared_count++;
    }
}
int main(){
    std::cout << "starting off..\n";
    auto start_time= std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for(int i = 0; i < 4; ++i) {
        threads.push_back(std::thread(worker));
    }
    for (auto& t : threads) {
        t.join();
    }
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Expected counter: " << 4*INCREMENTS_THREAD << "\n";
    std::cout << "Actual counter: " << shared_count << "\n";
    std::cout << "Time taken: " << duration.count() << " ms\n";
    return 0;
}

