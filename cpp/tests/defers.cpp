#define DEBUG
#include <core.hpp>

Task<Res<void>> defer_test_1(bool error) {
    // squential_defer;

    std::string prefix = error ? "[E] " : "[ ] ";

    defer {
        std::cout << prefix << "Defer 1" << std::endl;
    };

    defer_err {
        std::cout << prefix << "Defer 2 (err)" << std::endl;
    };

    defer_async {
        co_await waitMS(50);
        std::cout << prefix << "Defer 3 (async)" << std::endl;
    };

    defer_err_async {
        co_await waitMS(100);
        std::cout << prefix << "Defer 4 (async )" << std::endl;
    };

    defer {
        std::cout << prefix << "Defer 5 (sync after async)" << std::endl;
    };

    defer_err {
        std::cout << prefix << "Defer 6 (sync after async DEFER_ERR)" << std::endl;
    };

    if (error) {
        throw std::runtime_error("Error");
    }

    co_return_void;
};

Task<int> defer_tests () {
    std::cout << "===== Before defer_test_1 : no error" << std::endl;
    co_await defer_test_1(false);
    std::cout << "===== After defer_test_1 : no error" << std::endl;

    try_await(waitMS(150));
    std::cout << std::endl << std::endl;


    std::cout << "===== Before defer_test_1 : error" << std::endl;
    co_await defer_test_1(true);
    std::cout << "===== After defer_test_1 : error" << std::endl;

    try_await(waitMS(150));
    std::cout << std::endl << std::endl;

    co_return 0;
}

    // co_await defer_tests();
    // std::cout << "===== After defer_tests" << std::endl;



Task<int> d_async () {

    auto d = defer_single_scope([=]() -> Task<void> {
        const char buff[20] = { 65 };
        co_await waitMS(100);
        std::cout << "Defer async 2" << std::endl;
    }, false);

    co_return 1;
}

ErrTask init(int argc, char **argv) {

    co_await d_async();
    try_await(waitMS(1000));

    co_return_void;
}