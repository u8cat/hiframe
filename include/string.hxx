#pragma once

#include <iterator>
#include <string>
#include <cstdint>
#include <chrono>
#include <optional>

// Parse YYYY-MM-DDThh:mm:ss as a point in time in UTC, nothing if it is
// malformed. Both the EXIF date of a photo and the timestamp of a logo file name
// are written this way.
std::optional<std::chrono::sys_seconds> parse_datetime(const std::string &datetime);

// UTF-8 iterator
class utf8_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = char32_t;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const char32_t*;
    using reference         = const char32_t;

    char32_t operator*() const;
    utf8_iterator& operator++();

    utf8_iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    bool operator==(const utf8_iterator& other) const {
        return m_it == other.m_it;
    }

    bool operator!=(const utf8_iterator& other) const {
        return !(*this == other);
    }

    static utf8_iterator begin(const std::string &str) {
        return utf8_iterator(str.begin(), str.end());
    }

    static utf8_iterator end(const std::string &str) {
        return utf8_iterator(str.end(), str.end());
    }

protected:
    utf8_iterator(std::string::const_iterator it, std::string::const_iterator end)
        : m_it(it), m_end(end) {}

private:
    std::string::const_iterator m_it;
    std::string::const_iterator m_end;

    char32_t decode_n(int n, std::uint32_t cp) const;
};
