#include <cstdio>

#include "string.hxx"

std::optional<std::chrono::sys_seconds> parse_datetime(const std::string &datetime) {
    using namespace std::chrono;

    int y, mo, d, h, mi, s;
    if (sscanf(datetime.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &s) != 6)
        return {};

    auto date = year{y}/month{unsigned(mo)}/day{unsigned(d)};
    if (!date.ok()) return {};

    return sys_days(date) + hours{h} + minutes{mi} + seconds{s};
}

char32_t utf8_iterator::operator*() const {
    if (m_it == m_end) return 0;

    auto c = static_cast<unsigned char>(*m_it);
    uint32_t codepoint = 0;

    if ((c & 0x80) == 0)
        return static_cast<char32_t>(c);
    else if ((c & 0xE0) == 0xC0)
        return decode_n(1, c & 0x1F);
    else if ((c & 0xF0) == 0xE0)
        return decode_n(2, c & 0x0F);
    else if ((c & 0xF8) == 0xF0)
        return decode_n(3, c & 0x07);
    else
        return 0xFFFD; // Replacement Character
}

utf8_iterator& utf8_iterator::operator++() {
    if (m_it == m_end) return *this;

    auto c = static_cast<unsigned char>(*m_it);
    size_t len;

    if ((c & 0x80) == 0) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    else len = 1;

    for (size_t i = 0; i < len && m_it != m_end; ++i, ++m_it);
    return *this;
}

char32_t utf8_iterator::decode_n(int n, uint32_t cp) const {
    auto temp_it = m_it;

    for (int i = 0; i < n; ++i) {
        if (++temp_it == m_end) return 0xFFFD;

        unsigned char next_c = static_cast<unsigned char>(*temp_it);
        if ((next_c & 0xC0) != 0x80) return 0xFFFD;

        cp = (cp << 6) | (next_c & 0x3F);
    }
    return static_cast<char32_t>(cp);
}
