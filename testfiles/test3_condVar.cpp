#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

std::queue<int> q; std::mutex mtx; std::condition_variable cv;       
const int stopper = 0;             // "no more items"

void producer() {
    for (int i = 1; i <= 100; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        {   
            std::lock_guard<std::mutex> lock(mtx);
            q.push(i);
        }  

        cv.notify_one();  
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        q.push(stopper);}
    cv.notify_one();
}

void consumer() {
    while (true) {
        int item;
        {  
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [] { return !q.empty(); });
            std::cout << "(consumer):waiting for item\n";
            item = q.front();
            q.pop();
        }  

        if (item == stopper) {
            std::cout << "Ending program\n"; return;    
        }

        std::cout << "Processed item " << item << '\n';
    }
}


int main() {
    std::cout << "(main):opening the program.\n";
    std::thread producer_thread(producer);
    std::thread consumer_thread(consumer);
    producer_thread.join();
    consumer_thread.join();
    std::cout << "(main):Closing.\n";
    return 0;
}