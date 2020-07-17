// All the code in this file is adapted from Stack Overflow question:
// https://codereview.stackexchange.com/q/221626

// Licensed under CC BY-SA 4.0.
// https://creativecommons.org/licenses/by-sa/4.0/
//
// Question by osuka_:
// https://codereview.stackexchange.com/users/153926

// Answer by nwp & L. F.:
// https://codereview.stackexchange.com/users/47624
// https://codereview.stackexchange.com/users/188857

#pragma once

#include <condition_variable>
#include <functional> //bind
#include <future>     //packaged_task
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits> //invoke_result
#include <vector>

class thread_pool {
  public:
    thread_pool(size_t thread_count);
    ~thread_pool();

    // Since std::thread objects are not copiable, it doesn't make sense for a
    // thread_pool to be copiable.
    thread_pool(const thread_pool &) = delete;
    thread_pool &operator=(const thread_pool &) = delete;

    // F must be Callable, and invoking F with ...Args must be well-formed.
    template <typename F, typename... Args> auto execute(F &&, Args &&...);

  private:
    // _task_container_base and _task_container exist simply as a wrapper around
    // a MoveConstructible - but not CopyConstructible - Callable object.

    // _task_container_base exists only to serve as an abstract base for
    // _task_container.
    class _task_container_base {
      public:
        virtual ~_task_container_base(){};
        virtual void operator()() = 0;
    };

    //  _task_container takes a typename F, which must be Callable and
    //  MoveConstructible. Furthermore, F must be callable with no arguments; it
    //  can, for example, be a bind object with no placeholders. F may or may
    //  not be CopyConstructible.
    template <typename F> class _task_container : public _task_container_base {
      public:
        // Here, std::forward is needed because we need the construction of _f
        // *not* to bind an lvalue reference - it is not a guarantee that an
        // object of type F is CopyConstructible, only that it is
        // MoveConstructible.
        _task_container(F &&func) : _f(std::forward<F>(func)) {}
        void operator()() override { _f(); }

      private:
        F _f;
    };

    // Returns a unique_ptr to a _task_container that wraps around a given
    // function for details on _task_container_base and _task_container, see
    // above This exists so that _Func may be inferred from f.
    template <typename _Func>
    static std::unique_ptr<_task_container_base>
    allocate_task_container(_Func &&f) {
        // In the construction of the _task_container, f must be std::forward'ed
        // because it may not be CopyConstructible - the only requirement for an
        // instantiation of a _task_container is that the parameter is of a
        // MoveConstructible type.
        return std::unique_ptr<_task_container_base>(
            new _task_container<_Func>(std::forward<_Func>(f)));
    }

    std::vector<std::thread> _threads;
    std::queue<std::unique_ptr<_task_container_base>> _tasks;
    std::mutex _task_mutex;
    std::condition_variable _task_cv;
    bool _stop_threads = false;
};

thread_pool::thread_pool(size_t thread_count) {
    for (size_t i = 0; i < thread_count; ++i) {
        // Start waiting threads. Workers listen for changes through the
        // thread_pool member
        // condition_variablstd::future<std::invoke_result_t<F, Args...>>e
        _threads.emplace_back(std::thread([&]() {
            std::unique_lock<std::mutex> queue_lock(_task_mutex,
                                                    std::defer_lock);

            while (true) {
                queue_lock.lock();
                _task_cv.wait(queue_lock, [&]() -> bool {
                    return !_tasks.empty() || _stop_threads;
                });

                // Used by dtor to stop all threads without having to
                // unceremoniously stop tasks. The tasks must all be finished,
                // lest we break a promise and risk a future object throwing
                // an exception.
                if (_stop_threads && _tasks.empty())
                    return;

                // To initialize temp_task, we must move the unique_ptr from the
                // queue to the local stack. Since a unique_ptr cannot be
                // copied (obviously), it must be explicitly moved. This
                // transfers ownership of the pointed-to object to *this, as
                // specified in 20.11.1.2.1 [unique.ptr.single.ctor].
                auto temp_task = std::move(_tasks.front());

                _tasks.pop();
                queue_lock.unlock();

                (*temp_task)();
            }
        }));
    }
}

thread_pool::~thread_pool() {
    _stop_threads = true;
    _task_cv.notify_all();

    for (std::thread &thread : _threads) {
        thread.join();
    }
}

template <typename F, typename... Args>
auto thread_pool::execute(F &&function, Args &&... args) {
    std::unique_lock<std::mutex> queue_lock(_task_mutex, std::defer_lock);

    std::packaged_task<std::invoke_result_t<F, Args...>()> task_pkg(
        [f = std::forward<F>(function),
         largs = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            return std::apply(std::forward<F>(f), std::move(largs));
        });

    auto future = task_pkg.get_future();

    queue_lock.lock();

    // This lambda move-captures the packaged_task declared above. Since the
    // packaged_task type is not CopyConstructible, the function is not
    // CopyConstructible either - hence the need for a _task_container to wrap
    // around it.
    _tasks.emplace(allocate_task_container(std::move(task_pkg)));

    queue_lock.unlock();

    _task_cv.notify_one();

    return future;
}
