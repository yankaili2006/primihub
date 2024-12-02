//
// Created by scwang on 2023/12/26.
//

#ifndef CUSEAL_SAFEQUEUE_H
#define CUSEAL_SAFEQUEUE_H


#include <queue>
#include <mutex>
#include <condition_variable>
#include <initializer_list>

namespace util {

    template<typename T, size_t max_size = 0>
    class safe_queue {

    private:

        mutable std::mutex mut;
        mutable std::condition_variable push_cond;
        mutable std::condition_variable pop_cond;
        std::queue<T> data_queue;

    public:

        using value_type = typename std::queue<T>::value_type;
        using container_type = typename std::queue<T>::container_type;

        safe_queue() = default;

        safe_queue(const safe_queue &) = delete;

        safe_queue &operator=(const safe_queue &) = delete;

//        template<typename InputIterator>
//        safe_queue(InputIterator first, InputIterator last) {
//            for (auto iterator = first; iterator != last; ++iterator) {
//                data_queue.push(*iterator);
//            }
//        }
//
//        explicit safe_queue(const container_type &c) : data_queue(c) {}
//
//        safe_queue(std::initializer_list<value_type> list) : safe_queue(list.begin(), list.end()) {
//        }

        void push(const value_type &new_value) {
            if (max_size > 0) {
                std::unique_lock<std::mutex> lk(mut);
                push_cond.wait(lk, [this] { return this->data_queue.size() < max_size; });
                data_queue.push(std::move(new_value));
                lk.unlock();
            } else {
                std::lock_guard<std::mutex> lk(mut);
                data_queue.push(std::move(new_value));
            }
            pop_cond.notify_one();
        }

        value_type pop() {
            std::unique_lock<std::mutex> lk(mut);
            pop_cond.wait(lk, [this] { return !this->data_queue.empty(); });
            auto value = std::move(data_queue.front());
            data_queue.pop();
            lk.unlock();
            push_cond.notify_one();
            return value;
        }

        auto empty() const -> decltype(data_queue.empty()) {
            std::lock_guard<std::mutex> lk(mut);
            return data_queue.empty();
        }

        auto size() const -> decltype(data_queue.size()) {
            std::lock_guard<std::mutex> lk(mut);
            return data_queue.size();
        }
    };
}


#endif //CUSEAL_SAFEQUEUE_H
