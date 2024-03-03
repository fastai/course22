#pragma once

template <typename T>

class ScopeExit
{
public:
    ScopeExit(T t) : t(t) {}
    ~ScopeExit() { t(); }
    T t;
};

template <typename T>
ScopeExit<T> MoveScopeExit(T t) {
    return ScopeExit<T>(t);
};

#define NV_ANONYMOUS_VARIABLE_DIRECT(name, line) name##line
#define NV_ANONYMOUS_VARIABLE_INDIRECT(name, line) NV_ANONYMOUS_VARIABLE_DIRECT(name, line)

#define SCOPE_EXIT(func) const auto NV_ANONYMOUS_VARIABLE_INDIRECT(EXIT, __LINE__) = MoveScopeExit([=](){func;})