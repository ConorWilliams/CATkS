#pragma once

#include <vector>

template <typename T> class Cbuff {
  public:
    Cbuff(std::size_t size) : pos(0), max_size(size) {}

    bool contains(T const &) const;

    void push_back(T const &);
    void push_back(T &&);

    T pop_back();

    template <typename... Args> void emplace_back(Args &&...);

    std::size_t size() const { return buff.size(); }

    T &back() { return buff[pos]; };
    T const &back() const { return buff[pos]; };

    auto begin() { return buff.begin(); }
    auto begin() const { return buff.begin(); }

    auto end() { return buff.end(); }
    auto end() const { return buff.end(); }

  private:
    std::size_t pos;
    std::size_t max_size;
    std::vector<T> buff;
};

template <typename T> T Cbuff<T>::pop_back() {
    std::size_t tmp = pos;
    pos = (pos + max_size - 1) % max_size;
    return std::move(buff[tmp]);
}

template <typename T> bool Cbuff<T>::contains(T const &value) const {
    for (auto &&elem : buff) {
        if (value == elem) {
            return true;
        }
    }
    return false;
}

template <typename T> void Cbuff<T>::push_back(T const &value) {
    if (max_size != 0) {
        pos = (pos + 1) % max_size;

        if (buff.size() < max_size) {
            buff.push_back(value);
        } else {
            buff[pos] = value;
        }
    }
}

template <typename T> void Cbuff<T>::push_back(T &&value) {
    if (max_size != 0) {
        pos = (pos + 1) % max_size;

        if (buff.size() < max_size) {
            buff.push_back(std::move(value));
        } else {
            buff[pos] = std::move(value);
        }
    }
}

template <typename T>
template <typename... Args>
void Cbuff<T>::emplace_back(Args &&... args) {
    if (max_size != 0) {
        pos = (pos + 1) % max_size;

        if (buff.size() < max_size) {
            buff.emplace_back(std::forward<Args>(args)...);
        } else {
            buff[pos] = T(std::forward<Args>(args)...);
        }
    }
}
