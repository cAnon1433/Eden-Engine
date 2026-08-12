#include "ParticleThreadPool.h"

namespace Eden
{
    ParticleThreadPool::ParticleThreadPool(unsigned int threadCount)
    {
        unsigned int count = threadCount;
        if (count == 0)
        {
            count = std::thread::hardware_concurrency();
        }

        // hardware_concurrency() can return 0 if undetectable - falls
        // back to zero worker threads, which ParallelFor treats as
        // "always take the calling-thread fallback path" (see its
        // declaration comment in the header), not a crash.
        if (count == 0)
        {
            return;
        }

        m_LastSeenEpoch.assign(count, 0);
        m_Workers.reserve(count);
        for (unsigned int i = 0; i < count; ++i)
        {
            m_Workers.emplace_back(&ParticleThreadPool::WorkerLoop, this, static_cast<size_t>(i));
        }
    }

    ParticleThreadPool::~ParticleThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_Stop = true;
            ++m_Epoch;
        }
        m_WakeCV.notify_all();

        for (auto& worker : m_Workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    void ParticleThreadPool::ParallelFor(size_t count, const std::function<void(size_t, size_t)>& fn)
    {
        size_t threadCount = m_Workers.size();

        // Fallback path: no workers, or too little work for per-thread
        // dispatch/sync overhead to be worth it. threadCount <= 1 is
        // its own case, not just "small count": with only one worker,
        // there is no parallelism to gain at ANY particle count, only
        // condition-variable signaling cost to pay - so a 1-thread pool
        // (e.g. hardware_concurrency() == 1, or explicitly constructed
        // that way) always takes this path regardless of `count`. Not a
        // hypothetical case: this is exactly what running on a
        // single-core sandbox looks like, and mishandling it here would
        // make threading look like a regression on hardware where it
        // simply can't help.
        if (threadCount <= 1 || count < threadCount * 8)
        {
            fn(0, count);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_CurrentFn = fn;
            m_CurrentCount = count;
            m_PendingWorkers = threadCount;
            ++m_Epoch;
        }
        m_WakeCV.notify_all();

        std::unique_lock<std::mutex> lock(m_Mutex);
        m_DoneCV.wait(lock, [this] { return m_PendingWorkers == 0; });
    }

    void ParticleThreadPool::WorkerLoop(size_t workerIndex)
    {
        while (true)
        {
            std::function<void(size_t, size_t)> fn;
            size_t count = 0;

            {
                std::unique_lock<std::mutex> lock(m_Mutex);
                m_WakeCV.wait(lock, [this, workerIndex]
                {
                    return m_Stop || m_Epoch != m_LastSeenEpoch[workerIndex];
                });

                if (m_Stop)
                {
                    return;
                }

                m_LastSeenEpoch[workerIndex] = m_Epoch;
                fn = m_CurrentFn;
                count = m_CurrentCount;
            }

            size_t threadCount = m_Workers.size();
            size_t chunkSize = count / threadCount;
            size_t begin = workerIndex * chunkSize;
            size_t end = (workerIndex + 1 == threadCount) ? count : begin + chunkSize;

            if (begin < end)
            {
                fn(begin, end);
            }

            {
                std::lock_guard<std::mutex> lock(m_Mutex);
                --m_PendingWorkers;
                if (m_PendingWorkers == 0)
                {
                    m_DoneCV.notify_one();
                }
            }
        }
    }
}
