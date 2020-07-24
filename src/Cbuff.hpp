#pragma once

#include <vector>

template <typename T> class Cbuff {
  public:
    Cbuff(std::size_t size);

    bool contains(T const &) const;

    void push_back(T const &);
    void push_back(T &&);

    std::size_t size() const;

  private:
    std::size_t pos;
    std::size_t max_size;
    std::vector<T> buff;
};

template <typename T>
Cbuff<T>::Cbuff(std::size_t size) : pos(0), max_size(size) {}

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

template <typename T> std::size_t Cbuff<T>::size() const { return buff.size(); }
