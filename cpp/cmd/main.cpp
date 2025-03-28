#include <core.hpp>
#include <iostream>

// #include <torch/torch.h>

ErrTask init(int argc, char** argv) {

    // torch::Tensor tensor = torch::rand({3, 3});
    // std::cout << "Random Tensor:\n" << tensor << std::endl;
    
    while(true) {
        try_await(waitMS(1000));
        std::cout << "Hello World" << std::endl;
    }


    co_return_void;
}

