#include <chrono>    
#include <iostream>  
#include <string>    
#include <thread>    


void worker(const std::string& name) {
    for (int i = 1; i <= 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // one unit of "work"
        std::cout << name << " step " << i << '\n';
    }
    std::cout << name << " clocking out\n";
}

int main() {
    std::cout << "main: hiring two workers...\n";
    std::thread alice(worker, "A");
    std::thread bob(worker, "B");
    alice.join();
    bob.join();
    std::cout << "main: everyone is done, shutting down\n";
    return 0;
}