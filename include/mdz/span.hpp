#pragma once
// A minimal non-owning view over contiguous values, standing in for C++20's
// std::span. No member checks bounds.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mdz {

template <class T>
class Span {
public:
    constexpr Span() noexcept = default;
    constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}
    template <class U, class A>
    Span(std::vector<U, A>& v) noexcept : data_(v.data()), size_(v.size()) {}
    template <class U, class A>
    Span(const std::vector<U, A>& v) noexcept : data_(v.data()), size_(v.size()) {}

    constexpr T* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr T& operator[](std::size_t i) const noexcept { return data_[i]; }
    constexpr T* begin() const noexcept { return data_; }
    constexpr T* end() const noexcept { return data_ + size_; }
    constexpr Span subspan(std::size_t off, std::size_t n) const noexcept { return {data_ + off, n}; }
    constexpr Span first(std::size_t n) const noexcept { return {data_, n}; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

using ByteSpan = Span<const std::uint8_t>;

}  // namespace mdz
