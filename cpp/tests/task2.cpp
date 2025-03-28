// #define DEBUG
#include <core.hpp>



Task<int> t_imm() {
    co_return 1;
}

Task<int> t_susp() {
    co_await(waitMS(1));
    co_return 2;
}


ErrTask init(int argc, char **argv) {

    // case immediate.
    // 1 call immediate and don't wait;
    {

        // 1. Task::promise_type::promise_type
        // 2. Task::Task
        // 3. Task::promise_type::return_value
        // 4. Task::promise_type::final_suspend()
        // 5. Task::final_suspend_aw::final_suspend_aw
        // 6. Task::final_suspend_aw::await_ready
        // 7. await suspend indefinetly, becouse no continuation and no discard flag.
        
        // ~Task() should destroy the handle.

        t_imm();
    }

    // case wait for immediate.
    {
        // 1. Task::promise_type::promise_type
        // 2. Task::Task
        // 3. Task::promise_type::return_value
        // 4. Task::promise_type::final_suspend()
        // 5. Task::final_suspend_aw::final_suspend_aw
        // 6. Task::final_suspend_aw::await_ready
        // 7. await suspend indefinetly, becouse no continuation and no discard flag.

        // 8. Task::await_ready that return true.
        // 9. Task::await_resume 

        // ~Task() should destroy the handle.
        int res = co_await t_imm();
    }

    // case wait for suspended.
    {
        // ~Task() shouldn't destroy the handle and just set the discard flag.
        t_susp();
    }

    {
        // ~Task() should destroy the handle.
        int res = co_await t_susp();
    }

    co_return_void;
}