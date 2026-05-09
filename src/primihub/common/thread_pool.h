#ifndef SRC_PRIMIHUB_COMMON_THREAD_POOL_H_
#define SRC_PRIMIHUB_COMMON_THREAD_POOL_H_
#include <atomic>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

namespace primihub {

class ThreadPool {
 public:
  explicit ThreadPool(size_t size = std::thread::hardware_concurrency())
      : stop_(false) {
    for (size_t i = 0; i < size; i++) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] {
              return stop_ || !tasks_.empty();
            });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return result;
  }

  size_t size() const { return workers_.size(); }

  ~ThreadPool() {
    stop_ = true;
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::atomic<bool> stop_;
};

}  // namespace primihub
#endif  // SRC_PRIMIHUB_COMMON_THREAD_POOL_H_
