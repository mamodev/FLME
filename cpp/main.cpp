#include <core.hpp>
#include <iostream>


ErrTask init(int argc, char** argv) {
    
    while(true) {
        try_await(waitMS(1000));
        std::cout << "Hello World" << std::endl;
    }


    co_return_void;
}

