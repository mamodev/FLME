#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <iostream>
#include <coroutine>    

#include <vector>
#include <sstream>  
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>

#include "async.hpp"

template<typename T>
static std::vector<uint8_t> vec_serialize (T& data) {


    std::stringstream ss;
    {
        cereal::BinaryOutputArchive oarchive(ss);
        oarchive(data);
    }

    std::string serializedData = ss.str();
    std::vector<uint8_t> vec(serializedData.data(), serializedData.data() + serializedData.size());
    return vec;
}

template<typename T>
static T vec_deserialize (std::vector<uint8_t> data) {
    std::string serializedData(data.begin(), data.end());
    std::stringstream ss;
    ss.write(serializedData.data(), serializedData.size());


    T obj;
    {
        cereal::BinaryInputArchive iarchive(ss);
        iarchive(obj);
    }


    return obj;
}

namespace cereal {
    template <class Archive>
    void serialize(Archive& archive, Error& error) {
        archive(error.code, error.message);
    }

    template <class Archive, typename T>
    void save(Archive& archive, const Res<T>& res) {
        if (res.is_ok()) {
            archive(true);
            archive(res.getValue());
        }
        else {
            archive(false);
            archive(res.getError());
        }
    }

    template <class Archive, typename T>
    void load(Archive& archive, Res<T>& res) {
        bool is_ok;
        archive(is_ok);

        if (is_ok) {
            T value;
            archive(value);
            res = Res<T>(std::move(value));
        }
        else {
            Error error;
            archive(error);
            res = Res<T>(std::move(error));
        }
    }
}

// class RpcRemote {
// public:

//     RpcRemote() {}

//     RpcRemote(int socket) {
    
//     }

//     Task<std::vector<uint8_t>> call(uint16_t id, std::vector<uint8_t> data) {
//         co_return std::vector<uint8_t>();
//     }

//     static Task<RpcRemote> connect(std::string addr) {
//         co_return RpcRemote(0);
//     }

// private:
//     int socket;
// };

// template<typename A, typename R>
// class RpcInterface {
// public:
//     uint16_t id;
//     RpcInterface(uint16_t id) {
//         this->id = id;
//     }

//     Task<R> call(A& args, RpcRemote remote) {
//         std::vector<uint8_t> data = vec_serialize(args);
//         std::vector<uint8_t> res = co_await remote.call(this->id, data);
//         R result = vec_deserialize<R>(res);
//         co_return result;
//     }
// };


// auto CountRpc = RpcInterface<int, int>(1);
// using RpcRawHanlder = std::function<std::vector<uint8_t>(std::vector<uint8_t>)>;

// template<typename A, typename R>
// using RpcHandler = std::function<Res<R>(A&)>;

// class RpcServer {
// public:
//     std::map<uint16_t, RpcRawHanlder> handlers;

//     template<typename A, typename R>
//     void register_handler(uint16_t id, RpcHandler<A,R> fn) {
//         auto raw_handler = [fn](std::vector<uint8_t> data) -> std::vector<uint8_t> {
//             A args = vec_deserialize<A>(data);
//             Res<R> res = fn(args);

//             if (!res.is_ok()) {
//                 return vec_serialize<const char*>(res.getError());
//             }
//             else {
//                 return vec_serialize<R>(res.getValue());
//             }
//         };

//         this->handlers[id] = raw_handler;
//     }

//     std::vector<uint8_t> handle_request(uint16_t id, std::vector<uint8_t> data) {
//         return this->handlers[id](data);
//     }
// };


// Task<int> connect(std::string address, int port) {

//     int sock_fd = try_await(io::socket(AF_INET, SOCK_STREAM, 0, 0));
//     defer_error -> Task<int> {
//         co_await io::close_file(sock_fd);
//         co_return 0;
//     };

// }

Task<int> task (int i) {
    defer {
        std::cout << "Defer" << std::endl;
    };

    co_await io::waitMS(1000);

    if(i < 0) {
        co_return ERR_GENERIC;
    }

    co_return 0;
}


rutine m() {
    co_await task(0);
    
    co_return;
}

volatile bool running = 1;
void singal_handler(int sig) {
    running = 0;
}

int main () {   
    signal(SIGINT, singal_handler);
    signal(SIGTERM, singal_handler);
    int res = io::thread_initialize_async_engine();
    if (res < 0) {
        std::cerr << "io_uring_init failed" << std::endl;
        return 1;
    }   

    m();

    while ((running && corutines_active > 0) || !(io::io_uring.Idle())) {
        if(!running) {
            io::io_uring.cancel();
        }

        io::io_uring.handle_cqe();
    }
    return 0;
}



    


// template<typename A, typename R>
// class RpcFunction {
//     public:
//         std::function<Res<R>(A&)> fn;
//         RpcFunction(std::function<Res<R>(A&)> fn) {
//             this->fn = fn;
//         }

//         Res<R> operator()(A& args) {
//             return this->fn(args);
//         }
// };


// class RpcServer {
//     std::map<uint32_t, std::function<std::vector<uint8_t>(std::vector<uint8_t>)>> handlers;

//     public:
//         template<typename A, typename R>
//         void register_handler(uint32_t id, RpcFunction<A, R> fn) {
//             auto handler = [fn](std::vector<uint8_t> data) -> std::vector<uint8_t> {
//                 A args = vec_deserialize<A>(data);
//                 Res<R> res = fn(args);

//                 if (!res.is_ok()) {
//                     return vec_serialize<const char*>(res.getError());
//                 }
//                 else {
//                     return vec_serialize<R>(res.getValue());
//                 }
//             };


//             this->handlers[id] = handler;
//         }

//         std::vector<uint8_t> handle_request(uint32_t id, std::vector<uint8_t> data) {
//             return this->handlers[id](data);
//         }
// };
