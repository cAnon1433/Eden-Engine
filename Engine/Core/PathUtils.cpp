#include "PathUtils.h"

#include <stdexcept>
#include <vector>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#else
    #include <unistd.h>
    #include <climits>
#endif

namespace Eden
{
    namespace PathUtils
    {
        namespace
        {
            // Each platform branch resolves to the running executable's
            // own absolute path (not just its directory - callers below
            // strip the filename). All three retry-with-larger-buffer
            // rather than assuming a fixed size is always enough - long
            // usernames/install paths on Windows in particular can
            // exceed MAX_PATH (260 chars).
            std::filesystem::path QueryExecutablePath()
            {
#if defined(_WIN32)
                std::vector<wchar_t> buffer(MAX_PATH);
                for (;;)
                {
                    DWORD written = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
                    if (written == 0)
                    {
                        throw std::runtime_error("Eden: GetModuleFileNameW failed to resolve the executable path.");
                    }
                    // A truncated result fills the buffer exactly and
                    // sets ERROR_INSUFFICIENT_BUFFER - GetLastError is
                    // the only way to distinguish "exactly fit" from
                    // "truncated" since written == buffer.size() in
                    // both cases.
                    if (written < buffer.size() || GetLastError() != ERROR_INSUFFICIENT_BUFFER)
                    {
                        return std::filesystem::path(buffer.data(), buffer.data() + written);
                    }
                    buffer.resize(buffer.size() * 2);
                }
#elif defined(__APPLE__)
                uint32_t size = 0;
                _NSGetExecutablePath(nullptr, &size); // first call always fails, fills `size`
                std::vector<char> buffer(size);
                if (_NSGetExecutablePath(buffer.data(), &size) != 0)
                {
                    throw std::runtime_error("Eden: _NSGetExecutablePath failed to resolve the executable path.");
                }
                // May itself be a symlink (or contain one, e.g. inside
                // an .app bundle) - canonical() resolves it to a real
                // path so GetExecutableDir() is stable regardless of
                // how Eden was launched.
                return std::filesystem::canonical(std::filesystem::path(buffer.data()));
#else
                std::vector<char> buffer(PATH_MAX);
                for (;;)
                {
                    ssize_t written = readlink("/proc/self/exe", buffer.data(), buffer.size());
                    if (written < 0)
                    {
                        throw std::runtime_error("Eden: readlink(/proc/self/exe) failed to resolve the executable path.");
                    }
                    if (static_cast<size_t>(written) < buffer.size())
                    {
                        return std::filesystem::path(buffer.data(), buffer.data() + written);
                    }
                    buffer.resize(buffer.size() * 2);
                }
#endif
            }
        }

        const std::filesystem::path& GetExecutableDir()
        {
            // Function-local static: computed once on first call, reused
            // for the rest of the process's life - the executable
            // obviously doesn't move mid-run.
            static const std::filesystem::path dir = QueryExecutablePath().parent_path();
            return dir;
        }

        std::string ResolveResourcePath(const std::string& relativePath)
        {
            std::filesystem::path asPath(relativePath);
            if (asPath.is_absolute())
            {
                return relativePath;
            }
            return (GetExecutableDir() / asPath).string();
        }
    }
}
