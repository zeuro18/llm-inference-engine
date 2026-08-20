#include <future>
#include <iostream>
#include <thread>
#include <chrono>
void worker_function(std::promise<int> p) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p.set_value(123);
}

int main(){
    std::cout<< "Main:ordering..\n";

    std::promise<int> my_promise; //their half
    std::future<int> my_future =my_promise.get_future(); //our half

    std::thread t(worker_function, std::move(my_promise)); 
    
    auto start =std::chrono::steady_clock::now();
    std::cout<< "main: waiting..\n";
    int result=my_future.get();
    auto end= std::chrono::steady_clock::now();
    auto ms= std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "main: got " << result << " after waiting " << ms << " ms\n";
    t.join();
    return 0;
}

