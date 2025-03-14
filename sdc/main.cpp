
#include "asyncc.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


int main() {
    auto res = loop.init(1024);
    if(!res.is_ok()) {
        std::cerr << "Error initializing event loop: " << res.getError().message << std::endl;
        return 1;
    }


    loop.loop();
    return 0;
}