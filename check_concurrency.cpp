#include <thread>
#include <iostream>

int main()
{
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Number of concurrent threads supported: " << n << std::endl;
    return 0;
}
