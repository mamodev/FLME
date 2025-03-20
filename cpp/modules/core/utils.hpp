#include <concepts>
#include <coroutine>
#include <type_traits>

extern bool debug_enabled;

#ifdef DEBUG
#include <iostream>
#define debug(stmnt) if(debug_enabled) {std::cout << stmnt << std::endl;}
#else
#define debug(stmnt)
#endif


