#pragma once

#ifdef VKDNN_PROFILE
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#include <array>
#endif

namespace vkdnnProfile
{
inline std::string callerName(const std::string& name)
{
#ifdef _WIN32
    if (name != "api.free" && name != "api.reduce" && name != "api.memcpy.3") { return name; }
    std::array<void*, 24> frames{};
    CaptureStackBackTrace(2, static_cast<DWORD>(frames.size()), frames.data(), nullptr);
    static std::mutex mutex;
    static std::map<std::array<void*, 24>, std::string> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(frames);
    if (found != cache.end()) { return name + found->second; }
    static bool initialized = SymInitialize(GetCurrentProcess(), "C:/project/cccc/cccc/x64/Release", TRUE) != FALSE;
    std::string suffix;
    if (initialized)
    {
        for (auto address : frames)
        {
            if (!address) { break; }
            alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME]{};
            auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = MAX_SYM_NAME;
            DWORD64 displacement = 0;
            if (SymFromAddr(GetCurrentProcess(), reinterpret_cast<DWORD64>(address), &displacement, symbol))
            {
                suffix += "|";
                suffix += symbol->Name;
            }
        }
    }
    cache.emplace(frames, suffix);
    return name + suffix;
#else
    return name;
#endif
}
struct Entry
{
    unsigned long long calls = 0;
    double milliseconds = 0;
};
struct Statistics
{
    std::mutex mutex;
    std::map<std::string, Entry> entries;
    ~Statistics()
    {
        for (const auto& entry : entries)
        {
            std::fprintf(stderr, "VKPROFILE,%s,%llu,%.6f\n", entry.first.c_str(), entry.second.calls, entry.second.milliseconds);
        }
    }
};
inline Statistics& statistics()
{
    static Statistics value;
    return value;
}
struct Scope
{
    std::string name;
    std::chrono::steady_clock::time_point start;
    explicit Scope(std::string value) : name(callerName(value)), start(std::chrono::steady_clock::now()) { statistics(); }
    ~Scope()
    {
        double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        auto& state = statistics();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto& entry = state.entries[name];
        ++entry.calls;
        entry.milliseconds += milliseconds;
    }
};
}    //namespace vkdnnProfile
#define VKDNN_PROFILE_SCOPE(name) vkdnnProfile::Scope profileScope(name)
#else
#define VKDNN_PROFILE_SCOPE(name)
#endif