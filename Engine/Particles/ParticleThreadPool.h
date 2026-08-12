#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <cstdint>

namespace Eden
{
    // Minimal persistent thread pool for ParticleSystem's per-particle
    // parallel-for passes (density, forces, boundary collision) - NOT a
    // general-purpose task queue. Deliberately narrow: one operation
    // shape (ParallelFor over a contiguous [0, count) range, split into
    // contiguous chunks), because that's the only kind of parallelism
    // SPH's per-particle loops need. A general task-stealing pool would
    // be more code and more ways to get synchronization subtly wrong for
    // a benefit this module doesn't need.
    //
    // WHY NOT std::execution::par: Apple Clang's libc++ - what Eden
    // actually builds against on the primary Mac dev machine, see Eden
    // Doc - does not implement the C++17 Parallel STL execution
    // policies. Writing to std::execution::par here would either
    // silently compile as sequential or fail to compile outright,
    // depending on standard library - a real portability trap for a
    // project already juggling MoltenVK-specific quirks. A hand-rolled
    // pool is more code but behaves identically everywhere Eden builds.
    //
    // Threads are created ONCE at construction and parked between calls
    // (condition-variable wait), not spawned per ParallelFor call -
    // ParticleSystem::Step calls into this up to `substeps` times per
    // physics tick, up to several times per rendered frame. Thread
    // creation cost (OS-level allocation/scheduling setup) would eat
    // into or exceed the savings from parallelizing in the first place
    // if paid on every call.
    class ParticleThreadPool
    {
    public:
        // threadCount == 0 means std::thread::hardware_concurrency().
        // hardware_concurrency() is allowed to return 0 if it can't
        // determine the value - that case is NOT an error here, it just
        // means this pool ends up with zero worker threads, and
        // ParallelFor falls back to running everything on the calling
        // thread (see ParallelFor's comment) rather than crashing on a
        // zero-thread pool.
        explicit ParticleThreadPool(unsigned int threadCount = 0);
        ~ParticleThreadPool();

        // Non-copyable, non-movable - owns live OS threads with `this`
        // captured in their loop, same reasoning as most owning-thread
        // classes.
        ParticleThreadPool(const ParticleThreadPool&) = delete;
        ParticleThreadPool& operator=(const ParticleThreadPool&) = delete;

        // Splits [0, count) into ThreadCount() contiguous chunks (the
        // last chunk absorbs any remainder) and calls fn(begin, end) for
        // each chunk on a worker thread, blocking the calling thread
        // until every chunk completes.
        //
        // fn must be safe to call concurrently for DISJOINT [begin, end)
        // ranges - true for ParticleSystem's density/force/boundary
        // passes (each call only ever writes particle data at its own
        // index; reads of OTHER particles' data during those passes are
        // read-only - see ComputeDensityPressure/ComputeForces/
        // ResolveBoundaries, none of which mutate anything outside index
        // i's own slot).
        //
        // Falls back to calling fn(0, count) directly on the calling
        // thread - no worker dispatch at all - if there are zero usable
        // worker threads, or if `count` is small enough that per-thread
        // dispatch/sync overhead would likely dominate the actual work
        // (see the threshold in the .cpp). This keeps small particle
        // counts (a handful of particles right after a fresh Emit) from
        // paying parallel-dispatch cost for work that's cheaper to just
        // do inline.
        void ParallelFor(size_t count, const std::function<void(size_t begin, size_t end)>& fn);

        unsigned int ThreadCount() const { return static_cast<unsigned int>(m_Workers.size()); }

    private:
        void WorkerLoop(size_t workerIndex);

        std::vector<std::thread> m_Workers;

        std::mutex m_Mutex;
        std::condition_variable m_WakeCV; // workers wait on this for a new batch
        std::condition_variable m_DoneCV; // ParallelFor's caller waits on this for the batch to finish

        std::function<void(size_t, size_t)> m_CurrentFn;
        size_t m_CurrentCount = 0;

        // Bumped once per ParallelFor call. Each worker remembers the
        // last epoch it acted on (m_LastSeenEpoch) so a spurious wake
        // (or a wake left over from the previous batch) doesn't cause it
        // to redo stale work.
        uint64_t m_Epoch = 0;
        std::vector<uint64_t> m_LastSeenEpoch;

        size_t m_PendingWorkers = 0; // workers still working on the current batch
        bool m_Stop = false;
    };
}
