// The code in this file is adapted from Stack Overflow question:
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
#include <future> // packaged_task
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>       // apply
#include <type_traits> // invoke_result
#include <vector>

class ThreadPool {
  public:
    ThreadPool(size_t thread_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    template <typename F, typename... Args> auto execute(F &&, Args &&...);

  private:
    class TaskWrapBase {
      public:
        virtual ~TaskWrapBase(){};
        virtual void operator()() = 0;
    };

    template <typename F> class TaskWrap : public TaskWrapBase {
      public:
        TaskWrap(F &&func) : _f(std::forward<F>(func)) {}
        void operator()() override { _f(); }

      private:
        F _f;
    };

    std::vector<std::thread> _threads;
    std::queue<std::unique_ptr<TaskWrapBase>> _tasks;
    std::mutex _task_mutex;
    std::condition_variable _task_cv;
    bool _stop_threads = false;
};

ThreadPool::ThreadPool(size_t thread_count) {
    for (size_t i = 0; i < thread_count; ++i) {
        _threads.emplace_back(std::thread([&]() {
            std::unique_lock<std::mutex> queue_lock(_task_mutex,
                                                    std::defer_lock);

            while (true) {
                queue_lock.lock();

                _task_cv.wait(queue_lock, [&]() -> bool {
                    return !_tasks.empty() || _stop_threads;
                });

                if (_stop_threads && _tasks.empty()) {
                    return;
                }

                auto temp_task = std::move(_tasks.front());

                _tasks.pop();

                queue_lock.unlock();

                (*temp_task)();
            }
        }));
    }
}

ThreadPool::~ThreadPool() {
    _stop_threads = true;
    _task_cv.notify_all();

    for (std::thread &thread : _threads) {
        thread.join();
    }
}

template <typename F, typename... Args>
auto ThreadPool::execute(F &&function, Args &&... args) {
    std::unique_lock<std::mutex> queue_lock(_task_mutex, std::defer_lock);

    using pkg_t = std::packaged_task<std::invoke_result_t<F, Args...>()>;

    // std::packaged_task contains one-shot lambda
    pkg_t task_pkg(
        [f = std::forward<F>(function),
         largs = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            return std::apply(std::forward<F>(f), std::move(largs));
        });

    auto future = task_pkg.get_future();

    queue_lock.lock();

    _tasks.emplace(std::make_unique<TaskWrap<pkg_t>>(std::move(task_pkg)));

    queue_lock.unlock();

    _task_cv.notify_one();

    return future;
}
