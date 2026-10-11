#pragma once

// Eden's built-in profiler. Two halves:
//
//  * CPU sections - always on, ~20-30 ns per scope (one steady_clock read at
//    each end), so it is safe to leave in permanently. Wrap any block with
//    EDEN_PROFILE_SCOPE("name"). A section that runs several times in one
//    frame (e.g. each fixed physics step) is summed for that frame.
//
//  * GPU sections - filled in by Renderer from VkQueryPool timestamps, only
//    while Renderer::GpuTimingEnabled is on. Reported through SetGpu().
//    NOTE: on Apple GPUs (MoltenVK) timestamps are coarse/approximate - treat
//    those numbers as rough; the CPU "waiting on GPU" row is the reliable
//    GPU-bound indicator everywhere.
//
// Display: Eden::UI::DrawPerformancePanel. A pasteable text summary comes
// from BuildReport() (the panel's "Save report" button writes it to
// eden_perf.txt in the working directory).
//
// Single-threaded by design: everything that feeds it runs on the main thread.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace Eden
{
    class Profiler
    {
    public:
        using Clock = std::chrono::steady_clock;

        struct Entry
        {
            std::string name;
            bool gpu = false;
            float accum = 0.0f;   // ms added so far THIS frame
            float current = 0.0f; // ms last completed frame
            float avg = 0.0f;     // exponential moving average, ms
            float peak = 0.0f;    // slowly-decaying maximum, ms
            bool seen = false;
        };

        static constexpr int kHistory = 180;

        static Profiler& Get()
        {
            static Profiler instance;
            return instance;
        }

        static Clock::time_point Now() { return Clock::now(); }
        static float MsSince(Clock::time_point start)
        {
            return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
        }

        // Adds `ms` to the named section for the frame in progress.
        void Add(const char* name, float ms, bool gpu = false)
        {
            Entry* e = Find(name, gpu);
            e->accum += ms;
        }

        void SetGpu(const char* name, float ms) { Add(name, ms, true); }

        // Call once at the top of every frame with that frame's total
        // wall-clock time. Rolls the per-frame accumulators into the
        // averages/peaks and the frame-time history.
        void BeginFrame(float frameMs)
        {
            for (Entry& e : m_Entries)
            {
                e.current = e.accum;
                e.accum = 0.0f;
                if (!e.seen)
                {
                    e.avg = e.current;
                    e.peak = e.current;
                    e.seen = true;
                }
                else
                {
                    e.avg += (e.current - e.avg) * 0.05f;
                    e.peak = std::max(e.current, e.peak * 0.99f);
                }
            }

            // Slowest-recent-frame snapshot: averages smooth spikes away (a
            // 60 ms hitch on one click barely moves a 0.05-weighted average),
            // so remember the per-section breakdown of the worst frame seen
            // lately. The bar decays (~6 s half-life at 60 fps) so old spikes
            // eventually give way to newer ones.
            m_WorstFrameMs *= 0.998f;
            if (frameMs >= m_WorstFrameMs)
            {
                m_WorstFrameMs = frameMs;
                m_WorstSnapshot.clear();
                for (const Entry& e : m_Entries)
                {
                    if (e.current > 0.0f)
                    {
                        m_WorstSnapshot.push_back({ e.name + (e.gpu ? " [GPU]" : ""), e.current });
                    }
                }
                std::sort(m_WorstSnapshot.begin(), m_WorstSnapshot.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
            }

            m_FrameMs = frameMs;
            m_FrameAvg = (m_FrameCount == 0) ? frameMs : m_FrameAvg + (frameMs - m_FrameAvg) * 0.05f;
            m_FramePeak = std::max(frameMs, m_FramePeak * 0.99f);
            m_History[m_HistoryHead] = frameMs;
            m_HistoryHead = (m_HistoryHead + 1) % kHistory;
            ++m_FrameCount;
        }

        const std::vector<Entry>& Entries() const { return m_Entries; }
        float FrameMs() const { return m_FrameMs; }
        float FrameAvgMs() const { return m_FrameAvg; }
        float FramePeakMs() const { return m_FramePeak; }
        const float* History() const { return m_History.data(); }
        float WorstFrameMs() const { return m_WorstFrameMs; }
        // (section name, ms) for the slowest recent frame, biggest first.
        const std::vector<std::pair<std::string, float>>& WorstFrameBreakdown() const { return m_WorstSnapshot; }
        int HistoryOffset() const { return m_HistoryHead; }

        // Averaged ms of one named CPU section (0 if never recorded).
        float AvgOf(const char* name) const
        {
            for (const Entry& e : m_Entries)
            {
                if (!e.gpu && e.name == name)
                {
                    return e.avg;
                }
            }
            return 0.0f;
        }

        // Plain-text summary suitable for pasting into a bug report / chat.
        std::string BuildReport(const char* header = nullptr) const
        {
            std::string out;
            char line[256];
            if (header)
            {
                out += header;
                out += "\n";
            }
            const float avg = std::max(m_FrameAvg, 0.0001f);
            std::snprintf(line, sizeof(line), "Frame: avg %.2f ms (%.1f fps), worst recent %.2f ms\n", avg, 1000.0f / avg, m_FramePeak);
            out += line;

            const float waitGpu = AvgOf("Frame: waiting for GPU (fence)");
            const float waitVsync = AvgOf("Frame: acquire image / vsync");
            std::snprintf(line, sizeof(line), "Verdict: %s (GPU wait %.1f%% of frame, vsync/acquire %.1f%%)\n",
                          (waitGpu / avg > 0.45f) ? "GPU-BOUND" : "CPU-bound (or vsync-capped)", 100.0f * waitGpu / avg,
                          100.0f * waitVsync / avg);
            out += line;

            if (!m_WorstSnapshot.empty())
            {
                std::snprintf(line, sizeof(line), "\nSlowest recent frame: %.2f ms - where its time went (sections overlap; GPU rows lag 2 frames):\n", m_WorstFrameMs);
                out += line;
                int shown = 0;
                for (const auto& item : m_WorstSnapshot)
                {
                    if (item.second < 0.05f || shown++ >= 10) { break; }
                    std::snprintf(line, sizeof(line), "  %-52s %7.2f\n", item.first.c_str(), item.second);
                    out += line;
                }
            }

            out += "\nCPU sections (ms avg / recent worst):\n";
            for (const Entry& e : m_Entries)
            {
                if (!e.gpu)
                {
                    std::snprintf(line, sizeof(line), "  %-44s %7.2f / %7.2f\n", e.name.c_str(), e.avg, e.peak);
                    out += line;
                }
            }

            bool anyGpu = false;
            for (const Entry& e : m_Entries)
            {
                if (e.gpu)
                {
                    if (!anyGpu)
                    {
                        out += "\nGPU passes (ms avg / recent worst; approximate on Apple GPUs):\n";
                        anyGpu = true;
                    }
                    std::snprintf(line, sizeof(line), "  %-44s %7.2f / %7.2f\n", e.name.c_str(), e.avg, e.peak);
                    out += line;
                }
            }
            return out;
        }

        bool SaveReport(const char* path, const char* header = nullptr) const
        {
            FILE* f = std::fopen(path, "w");
            if (!f)
            {
                return false;
            }
            const std::string text = BuildReport(header);
            std::fwrite(text.data(), 1, text.size(), f);
            std::fclose(f);
            return true;
        }

    private:
        Entry* Find(const char* name, bool gpu)
        {
            for (Entry& e : m_Entries)
            {
                if (e.gpu == gpu && e.name == name)
                {
                    return &e;
                }
            }
            m_Entries.emplace_back();
            m_Entries.back().name = name;
            m_Entries.back().gpu = gpu;
            return &m_Entries.back();
        }

        std::vector<Entry> m_Entries;
        std::vector<std::pair<std::string, float>> m_WorstSnapshot;
        float m_WorstFrameMs = 0.0f;
        std::array<float, kHistory> m_History{};
        int m_HistoryHead = 0;
        float m_FrameMs = 0.0f;
        float m_FrameAvg = 0.0f;
        float m_FramePeak = 0.0f;
        unsigned long long m_FrameCount = 0;
    };

    // RAII CPU timer: adds its lifetime to the named section.
    class ProfileScope
    {
    public:
        explicit ProfileScope(const char* name) : m_Name(name), m_Start(Profiler::Now()) {}
        ~ProfileScope() { Profiler::Get().Add(m_Name, Profiler::MsSince(m_Start)); }
        ProfileScope(const ProfileScope&) = delete;
        ProfileScope& operator=(const ProfileScope&) = delete;

    private:
        const char* m_Name;
        Profiler::Clock::time_point m_Start;
    };
}

#define EDEN_PROFILE_CONCAT_INNER(a, b) a##b
#define EDEN_PROFILE_CONCAT(a, b) EDEN_PROFILE_CONCAT_INNER(a, b)
#define EDEN_PROFILE_SCOPE(name) ::Eden::ProfileScope EDEN_PROFILE_CONCAT(edenProfileScope_, __LINE__)(name)
