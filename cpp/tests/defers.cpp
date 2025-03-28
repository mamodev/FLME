// #define DEBUG
#include <core.hpp>

// Now that promise_type is fully defined, we define defer_manager
c_defer defer_s() {
    auto h = get_current_handle();
    std::cout << "SYN, executing: " << h.promise().ID << std::endl;
    co_return;
}

c_defer defer_as() {
    co_await waitMS(1);
    auto h = get_current_handle();
    std::cout << "AS, executing: " << h.promise().ID << std::endl;   
     co_return;
}

Task<Res<int>> p() {
    co_await std::suspend_always{};

    std::cout << "this is an example of continuation" << std::endl;

    co_return 1;
}

ErrTask init(int argc, char **argv) {
    // creturn;

    // auto ph = p();
    // auto dm = c_defer::defer_manager();

    // {
    //     auto dh1 = defer_s();
    //     auto dh2 = defer_as();
    //     auto dh3 = [&]() -> c_defer {
    //         // co_await waitMS(1);

    //         auto h = get_current_handle();
    //         std::cout << "Y, executing: " << h.promise().ID << std::endl;
            
    //         co_return;
    //     }();


       
    //     dh1.h.promise().dm = &dm;
    //     dh2.h.promise().dm = &dm;
    //     dh3.h.promise().dm = &dm;
    //     dm.h = ph.handle;
    // }

    // std::cout << "===== Before ph" << std::endl;
    // co_await ph;

    // try_await(waitMS(1000));
    co_return_void;
}

    // co_await d_async();
    // try_await(waitMS(1000));

    // Task<Res<void>> defer_test_1(bool error) {
    //     // squential_defer;
    
    //     std::string prefix = error ? "[E] " : "[ ] ";
    
    //     defer {
    //         std::cout << prefix << "Defer 1" << std::endl;
    //     };
    
    //     defer_err {
    //         std::cout << prefix << "Defer 2 (err)" << std::endl;
    //     };
    
    //     defer_async {
    //         co_await waitMS(50);
    //         std::cout << prefix << "Defer 3 (async)" << std::endl;
    //     };
    
    //     defer_err_async {
    //         co_await waitMS(100);
    //         std::cout << prefix << "Defer 4 (async )" << std::endl;
    //     };
    
    //     defer {
    //         std::cout << prefix << "Defer 5 (sync after async)" << std::endl;
    //     };
    
    //     defer_err {
    //         std::cout << prefix << "Defer 6 (sync after async DEFER_ERR)" << std::endl;
    //     };
    
    //     if (error) {
    //         throw std::runtime_error("Error");
    //     }
    
    //     co_return_void;
    // };
    
    // Task<int> defer_tests () {
    //     std::cout << "===== Before defer_test_1 : no error" << std::endl;
    //     co_await defer_test_1(false);
    //     std::cout << "===== After defer_test_1 : no error" << std::endl;
    
    //     try_await(waitMS(150));
    //     std::cout << std::endl << std::endl;
    
    
    //     std::cout << "===== Before defer_test_1 : error" << std::endl;
    //     co_await defer_test_1(true);
    //     std::cout << "===== After defer_test_1 : error" << std::endl;
    
    //     try_await(waitMS(150));
    //     std::cout << std::endl << std::endl;
    
    //     co_return 0;
    // }
    
    //     // co_await defer_tests();
    //     // std::cout << "===== After defer_tests" << std::endl;
    
    
    
    // Task<int> d_async () {
    
    //     auto d = defer_single_scope([=]() -> Task<void> {
    //         const char buff[20] = { 65 };
    //         co_await waitMS(100);
    //         std::cout << "Defer async 2" << std::endl;
    //     }, false);
    
    //     co_return 1;
    // }
    