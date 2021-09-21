#pragma once

#include <list>
#include <set>
#include <string>
#include <sstream>

template <typename T>
constexpr void multiset_sum(std::list<std::multiset<T>> &ret,
                  const std::multiset<T> &incomplete_multiset, T n,
                  const std::set<T, std::greater<T>> &values) {
    auto smaller_values = values;
    for (T next : values) {
        if (next > n) {
            continue;
        }
        auto multiset = incomplete_multiset;
        multiset.insert(next);
        if (next == n) {
            ret.push_back(multiset);
            smaller_values.erase(next);
            continue;
        }
        multiset_sum(ret, multiset, n - next, smaller_values);
        smaller_values.erase(next);
    }
}

template <typename T>
constexpr std::list<std::multiset<T>>
multiset_sum(T n, const std::set<T, std::greater<T>> &values) {
    std::list<std::multiset<T>> ret;
    multiset_sum(ret, {}, n, values);
    return ret;
}

template <typename T, typename U>
std::string multiset_to_string(const std::multiset<T, U> &s) {
    std::stringstream ss;
    for (const auto &element : s) {
        ss << element << ", ";
    }
    return ss.str();
}
