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
#include <future> //packaged_task
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits> //invoke_result
#include <vector>

class _task_container_base {
  public:
    virtual ~_task_container_base(){};
    virtual void operator()() = 0;
};

template <typename F> class _task_container : public _task_container_base {
  public:
    _task_container(F &&func) : _f(std::forward<F>(func)) {}
    void operator()() override { _f(); }

  private:
    F _f;
};

template <typename F> _task_container(F &&)->_task_container<F>;

class thread_pool {
  public:
    thread_pool(size_t thread_count);
    ~thread_pool();

    thread_pool(const thread_pool &) = delete;
    thread_pool &operator=(const thread_pool &) = delete;

    template <typename F, typename... Args> auto execute(F &&, Args &&...);

  private:
    std::vector<std::thread> _threads;
    std::queue<std::unique_ptr<_task_container_base>> _tasks;
    std::mutex _task_mutex;
    std::condition_variable _task_cv;
    bool _stop_threads = false;
};

thread_pool::thread_pool(size_t thread_count) {
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

    _tasks.emplace(new _task_container(std::move(task_pkg)));

    queue_lock.unlock();

    _task_cv.notify_one();

    return future;
}
