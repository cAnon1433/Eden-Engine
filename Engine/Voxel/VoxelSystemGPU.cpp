#include "VoxelSystemGPU.h"
#include "../Core/Profiler.h"
#include <cstdlib>
#include <cmath>
#include "MarchingCubesTables33.h"
#include "../Renderer/Vulkan/RendererTypes.h"
#include "../Physics/SDF.h"

#include <glm/gtc/noise.hpp> // glm::perlin - used by SeedHeightfieldNoise only; already vendored, no new ThirdParty dependency (GTC noise is stable, no GLM_ENABLE_EXPERIMENTAL needed)

#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <queue>

namespace Eden
{
    namespace
    {
        constexpr uint32_t kLocalSizeX = 64;

        uint32_t DispatchGroupCount(uint32_t itemCount)
        {
            return (itemCount + kLocalSizeX - 1) / kLocalSizeX;
        }

        // Polynomial smooth minimum (Inigo Quilez's formula) - blends
        // two SDF values with a rounded fillet instead of the hard
        // corner a plain std::min leaves at the boundary between two
        // shapes. k controls the blend radius: 0 degenerates to a plain
        // min (sharp seams), larger k rounds the seam more but also
        // starts eating into real surface detail if particles that
        // AREN'T touching get pulled together. Used by
        // SeedFromParticles to fix the "visible individual spheres"
        // look a hard union produces - each particle's dome stayed
        // visible even where two spheres overlapped, since min() just
        // picks whichever surface is closer rather than blending them.
        float SmoothMin(float a, float b, float k)
        {
            if (k <= 0.0f)
            {
                return std::min(a, b);
            }
            float h = std::clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
            return glm::mix(b, a, h) - k * h * (1.0f - h);
        }

        // cell -> chunk coordinate (not flat index), clamped to the
        // volume's actual chunk grid - used by Carve() to turn a
        // world-space AABB into an inclusive chunk-coordinate range to
        // iterate. Separate from VoxelSystemGPU::ChunkIndexForCell
        // (which returns the flat index the shader/dirty-flag buffer
        // use) because Carve() needs the 3D coordinate to loop a range,
        // not a single flat index.
        glm::ivec3 ChunkCoordFromCellClamped(const Eden::VoxelVolumeDesc& desc, glm::ivec3 cell)
        {
            glm::ivec3 coord = cell / static_cast<int32_t>(Eden::kVoxelChunkSize);
            return glm::clamp(coord, glm::ivec3(0), desc.chunkDims - glm::ivec3(1));
        }

        // Vulkan's own indirect-draw command layout - not an Eden type,
        // deliberately matching VkDrawIndirectCommand field-for-field so
        // the buffer this class builds is exactly what vkCmdDrawIndirect
        // expects with zero translation. Named locally rather than just
        // using VkDrawIndirectCommand directly only so InitialCommands()
        // below reads clearly about which field is which.
        struct IndirectCommand
        {
            uint32_t vertexCount;
            uint32_t instanceCount;
            uint32_t firstVertex;
            uint32_t firstInstance;
        };
        static_assert(sizeof(IndirectCommand) == sizeof(VkDrawIndirectCommand), "must match Vulkan's own struct layout");
    }

    void VoxelSystemGPU::Init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                               VkCommandPool commandPool, VkQueue graphicsQueue)
    {
        m_Device = device;
        m_Allocator = allocator;
        m_CommandPool = commandPool;
        m_GraphicsQueue = graphicsQueue;

        // Created UNSIGNALED - SubmitAndWaitForCommandBuffer's first use
        // submits into it directly; vkResetFences before every
        // subsequent submission is what makes reuse safe (a fence must
        // be reset between the previous wait and the next submit, or the
        // next vkWaitForFences would just immediately return on the
        // OLD signal).
        VkFenceCreateInfo submitFenceInfo{};
        submitFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(m_Device, &submitFenceInfo, nullptr, &m_SubmitFence) != VK_SUCCESS)
        {
            // Non-fatal - SubmitAndWaitForCommandBuffer falls back to the
            // old vkQueueWaitIdle path whenever m_SubmitFence is
            // VK_NULL_HANDLE, same as its per-call fence-creation failure
            // path already did before this became a persistent fence.
            m_SubmitFence = VK_NULL_HANDLE;
        }

        // Queried here (previously the physicalDevice parameter was
        // unused) - minStorageBufferOffsetAlignment is what every
        // volume's densityOffsetElements (see RegisterVolume) must
        // respect when used as a VkDescriptorBufferInfo::offset into
        // m_SharedDensityBuffer, both for voxel_march.comp's per-volume
        // compute bind and, indirectly, for RaymarchObjectGPU::
        // densityOffset's correctness (that one is consumed inside a
        // shader as a plain array index, not a descriptor offset, so it
        // isn't itself alignment-constrained by Vulkan - but keeping
        // every volume's offset on this same alignment grid regardless
        // is simpler than tracking two different offset conventions for
        // the same underlying data).
        VkPhysicalDeviceProperties deviceProps{};
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProps);
        m_StorageBufferOffsetAlignmentBytes = deviceProps.limits.minStorageBufferOffsetAlignment;
        if (m_StorageBufferOffsetAlignmentBytes == 0)
        {
            m_StorageBufferOffsetAlignmentBytes = 1; // spec allows 1 as a valid (if unusual) minimum
        }

        // See m_SharedDensityBuffer's own comment for the full
        // rationale (bindless was tried and found unsupported on this
        // project's dev hardware; one shared buffer + per-volume
        // offsets is the portable fix). Allocated once here, up front,
        // at kSharedDensityBufferCapacityElements - RegisterVolume
        // bump-allocates from this, never grows it.
        m_SharedDensityBufferCapacityElements = kSharedDensityBufferCapacityElements;
        m_SharedDensityBufferUsedElements = 0;
        m_SharedDensityBuffer.Init(m_Allocator, m_SharedDensityBufferCapacityElements * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Uploaded once, read-only forever after - every volume's
        // compute descriptor set points at this SAME buffer (see
        // CreateComputeLayout's per-volume descriptor writes).
        m_MC33Table.Init(m_Allocator, sizeof(kMC33TableData),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        UploadRange(m_MC33Table.Get(), 0, kMC33TableData, sizeof(kMC33TableData));

        CreateComputeLayout();
        CreateComputePipelines();

        m_Volumes.reserve(kVoxelMaxVolumes);
        m_Generations.reserve(kVoxelMaxVolumes);
    }

    void VoxelSystemGPU::CreateComputeLayout()
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        // 0 = density, 1 = vertices (out), 2 = indirect commands (atomic
        // vertexCount), 3 = MC33 table blob, 4 = dirty flags, 5 = chunk
        // remap (raw chunk index -> compact vertex/indirect slot, or
        // kVoxelChunkRemapUnused - see ChunkRemapBuffer's own comment in
        // voxel_march.comp) - all read by voxel_march.comp; density is
        // written from the CPU side only now (UploadDensity), never by a
        // compute shader - see the class comment in VoxelSystemGPU.h.
        // (Was 6 bindings, edge table + tri table separately, before the
        // MC33 rewrite packed every table into one buffer - see
        // MarchingCubesTables33.h. Back to 6 again now, for an unrelated
        // reason - chunk compaction, not the old tri-table split.)

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ComputeSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create VoxelSystemGPU compute descriptor set layout");
        }

        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kVoxelMaxVolumes * static_cast<uint32_t>(bindings.size()) },
        };
        m_ComputeDescriptorPool.Init(m_Device, poolSizes, kVoxelMaxVolumes);
    }

    void VoxelSystemGPU::CreateComputePipelines()
    {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(VoxelParamsGPU);

        m_ComputePipelineLayout.Init(m_Device, { m_ComputeSetLayout }, { pushConstantRange });

        // The only remaining GPU compute pipeline - see the class
        // comment in VoxelSystemGPU.h on why seeding/carving moved to
        // plain CPU code and no longer need their own.
        m_MarchPipeline.Init(m_Device, m_ComputePipelineLayout.Get(), "Shaders/Compiled/voxel_march.comp.spv");
    }

    void VoxelSystemGPU::Shutdown()
    {
        if (m_SubmitFence != VK_NULL_HANDLE)
        {
            vkDestroyFence(m_Device, m_SubmitFence, nullptr);
            m_SubmitFence = VK_NULL_HANDLE;
        }

        if (m_DensityStagingBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_Allocator, m_DensityStagingBuffer, m_DensityStagingAllocation);
            m_DensityStagingBuffer = VK_NULL_HANDLE;
            m_DensityStagingAllocation = VK_NULL_HANDLE;
            m_DensityStagingMappedPtr = nullptr;
            m_DensityStagingCapacityBytes = 0;
        }

        if (m_ReusableCommandBuffer != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &m_ReusableCommandBuffer);
            m_ReusableCommandBuffer = VK_NULL_HANDLE;
        }

        for (auto& volume : m_Volumes)
        {
            volume.instance.Shutdown();
            volume.dirtyFlags.Shutdown();
            volume.chunkRemap.Shutdown();
            volume.indirect.Shutdown();
            volume.vertices.Shutdown();
            // No per-volume volume.density.Shutdown() here anymore -
            // Volume no longer owns a density buffer (see
            // m_SharedDensityBuffer's comment); it's shut down once,
            // below, after this loop, instead of once per volume.
        }
        m_Volumes.clear();
        m_Generations.clear();
        m_FreeSlotIndices.clear();
        m_FreeComputeSets.clear(); // pool Shutdown() below invalidates these handles anyway
        m_FreeDensityRegions.clear();

        m_SharedDensityBuffer.Shutdown();

        m_MarchPipeline.Shutdown();
        m_ComputePipelineLayout.Shutdown();
        m_ComputeDescriptorPool.Shutdown();

        if (m_ComputeSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_Device, m_ComputeSetLayout, nullptr);
            m_ComputeSetLayout = VK_NULL_HANDLE;
        }

        m_MC33Table.Shutdown();
    }

    // Identical shape to ParticleSystemGPU::UploadRange - see that
    // class's copy for the full reasoning (blocking staged upload,
    // fine for infrequent/registration-time calls, not per-frame).
    void VoxelSystemGPU::UploadRange(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size)
    {
        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: VoxelSystemGPU failed to create staging buffer");
        }

        std::memcpy(stagingInfoOut.pMappedData, data, static_cast<size_t>(size));

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandPool = m_CommandPool;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, stagingBuffer, dst, 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_GraphicsQueue);

        vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &cmd);
        vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
    }

    int32_t VoxelSystemGPU::ChunkIndexForCell(const VoxelVolumeDesc& desc, glm::ivec3 cell)
    {
        glm::ivec3 chunkCoord = cell / static_cast<int32_t>(kVoxelChunkSize);
        chunkCoord = glm::clamp(chunkCoord, glm::ivec3(0), desc.chunkDims - glm::ivec3(1));
        return chunkCoord.x + chunkCoord.y * desc.chunkDims.x + chunkCoord.z * desc.chunkDims.x * desc.chunkDims.y;
    }

    uint32_t VoxelSystemGPU::SampleIndex(const VoxelVolumeDesc& desc, glm::ivec3 sample)
    {
        glm::ivec3 dims = desc.SampleDims();
        glm::ivec3 clamped = glm::clamp(sample, glm::ivec3(0), dims - glm::ivec3(1));
        return static_cast<uint32_t>(clamped.x) + static_cast<uint32_t>(clamped.y) * static_cast<uint32_t>(dims.x)
             + static_cast<uint32_t>(clamped.z) * static_cast<uint32_t>(dims.x) * static_cast<uint32_t>(dims.y);
    }

    float VoxelSystemGPU::SampleDensityTrilinear(const Volume& volume, glm::vec3 localPos)
    {
        glm::vec3 gridPos = localPos / volume.desc.voxelSize;
        glm::ivec3 sampleDims = volume.desc.SampleDims();

        // Clamp so the 8-corner fetch below never reaches for i0+1
        // beyond the last valid sample - SampleIndex() itself also
        // clamps per-corner, so this is a belt-and-suspenders bound on
        // the interpolation factor, not strictly required for safety,
        // but keeps frac in [0,1) as intended rather than extrapolating.
        glm::vec3 maxGrid = glm::vec3(sampleDims) - glm::vec3(1.0f + 1e-4f);
        glm::vec3 clampedGrid = glm::clamp(gridPos, glm::vec3(0.0f), glm::max(maxGrid, glm::vec3(0.0f)));

        glm::ivec3 i0 = glm::ivec3(glm::floor(clampedGrid));
        glm::vec3 frac = clampedGrid - glm::vec3(i0);

        float c000 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(0, 0, 0))];
        float c100 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(1, 0, 0))];
        float c010 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(0, 1, 0))];
        float c110 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(1, 1, 0))];
        float c001 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(0, 0, 1))];
        float c101 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(1, 0, 1))];
        float c011 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(0, 1, 1))];
        float c111 = volume.densityCPU[SampleIndex(volume.desc, i0 + glm::ivec3(1, 1, 1))];

        float c00 = glm::mix(c000, c100, frac.x);
        float c10 = glm::mix(c010, c110, frac.x);
        float c01 = glm::mix(c001, c101, frac.x);
        float c11 = glm::mix(c011, c111, frac.x);
        float c0 = glm::mix(c00, c10, frac.y);
        float c1 = glm::mix(c01, c11, frac.y);
        return glm::mix(c0, c1, frac.z);
    }

    void VoxelSystemGPU::UploadDensity(Volume& volume)
    {
        // Whole-field upload - only appropriate when every sample
        // actually changed (SeedSphere). See UploadDensityRegion for
        // the partial-region path Carve() uses instead. Destination
        // offset is this volume's region WITHIN the shared buffer
        // (densityOffsetElements), not absolute 0 - see
        // m_SharedDensityBuffer's comment for why every volume now
        // shares one buffer instead of owning its own.
        UploadRange(m_SharedDensityBuffer.Get(), volume.densityOffsetElements * sizeof(float),
                    volume.densityCPU.data(), volume.densityCPU.size() * sizeof(float));
    }

    void VoxelSystemGPU::EnsureDensityStagingCapacity(VkDeviceSize requiredBytes)
    {
        if (requiredBytes <= m_DensityStagingCapacityBytes)
        {
            return; // common case - existing buffer already has enough room, zero allocation this call
        }

        if (m_DensityStagingBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_Allocator, m_DensityStagingBuffer, m_DensityStagingAllocation);
            m_DensityStagingBuffer = VK_NULL_HANDLE;
            m_DensityStagingAllocation = VK_NULL_HANDLE;
            m_DensityStagingMappedPtr = nullptr;
            m_DensityStagingCapacityBytes = 0;
        }

        // Grow with headroom (doubling), not exactly-to-fit - a slightly
        // bigger next carve (e.g. a wider drill-through stroke) shouldn't
        // immediately trigger ANOTHER reallocation right after this one.
        VkDeviceSize newCapacity = std::max<VkDeviceSize>(requiredBytes, m_DensityStagingCapacityBytes * 2);

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = newCapacity;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &m_DensityStagingBuffer,
                             &m_DensityStagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: VoxelSystemGPU failed to (re)create the persistent density staging buffer");
        }

        m_DensityStagingMappedPtr = stagingInfoOut.pMappedData;
        m_DensityStagingCapacityBytes = newCapacity;
    }

    VkCommandBuffer VoxelSystemGPU::GetReusableCommandBuffer()
    {
        if (m_ReusableCommandBuffer == VK_NULL_HANDLE)
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandPool = m_CommandPool;
            cmdAllocInfo.commandBufferCount = 1;
            vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &m_ReusableCommandBuffer);
        }
        else
        {
            // Explicit reset rather than relying on vkBeginCommandBuffer's
            // implicit reset - both are valid given
            // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT (see
            // m_ReusableCommandBuffer's own comment), but being explicit
            // here doesn't depend on that implicit-reset behavior staying
            // true if this pool's creation flags ever change for some
            // unrelated reason.
            vkResetCommandBuffer(m_ReusableCommandBuffer, 0);
        }
        return m_ReusableCommandBuffer;
    }

    void VoxelSystemGPU::UploadDensityRegion(Volume& volume, glm::ivec3 sampleMin, glm::ivec3 sampleMax)
    {
        EDEN_PROFILE_SCOPE("Voxel: density upload (submit + GPU wait)");
        // Row-by-row (fixed y,z, contiguous run of x) is the natural
        // packing given SampleIndex's x-fastest layout - each row is
        // already contiguous in densityCPU, so it can go straight into
        // one VkBufferCopy region without further splitting.
        uint32_t rowLen = static_cast<uint32_t>(sampleMax.x - sampleMin.x + 1);
        uint32_t rowCount = static_cast<uint32_t>((sampleMax.y - sampleMin.y + 1) * (sampleMax.z - sampleMin.z + 1));
        VkDeviceSize rowBytes = static_cast<VkDeviceSize>(rowLen) * sizeof(float);
        VkDeviceSize totalBytes = rowBytes * rowCount;

        // Every dstOffset below is measured from this volume's own
        // region start within the shared buffer, not absolute 0 - see
        // UploadDensity's comment/m_SharedDensityBuffer's own comment.
        VkDeviceSize volumeBaseBytes = volume.densityOffsetElements * sizeof(float);

        // Grows the persistent buffer only if this carve's region is
        // bigger than anything seen so far this session - see
        // EnsureDensityStagingCapacity's own comment. No per-call
        // vmaCreateBuffer/vmaDestroyBuffer anymore, and no per-call
        // `packed` heap vector either - both replaced by writing
        // straight into the persistent buffer's own (already mapped,
        // never re-mapped) memory below.
        EnsureDensityStagingCapacity(totalBytes);

        std::vector<VkBufferCopy> regions;
        regions.reserve(rowCount);

        size_t packedOffsetElems = 0;
        char* stagingWritePtr = static_cast<char*>(m_DensityStagingMappedPtr);
        for (int32_t z = sampleMin.z; z <= sampleMax.z; ++z)
        for (int32_t y = sampleMin.y; y <= sampleMax.y; ++y)
        {
            uint32_t rowStartIdx = SampleIndex(volume.desc, glm::ivec3(sampleMin.x, y, z));
            std::memcpy(stagingWritePtr + packedOffsetElems * sizeof(float), &volume.densityCPU[rowStartIdx],
                        static_cast<size_t>(rowBytes));

            VkBufferCopy region{};
            region.srcOffset = packedOffsetElems * sizeof(float);
            region.dstOffset = volumeBaseBytes + static_cast<VkDeviceSize>(rowStartIdx) * sizeof(float);
            region.size = rowBytes;
            regions.push_back(region);

            packedOffsetElems += rowLen;
        }

        // Same shape as UploadRange, but with MULTIPLE copy regions in
        // one vkCmdCopyBuffer call instead of one region per call - the
        // whole point of batching this: one blocking submit for the
        // entire touched region, not one per row (which, for a several-
        // dozen-row carve region, would cost more in submit/wait
        // overhead than the whole-buffer upload this replaces).
        VkCommandBuffer cmd = GetReusableCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        vkCmdCopyBuffer(cmd, m_DensityStagingBuffer, m_SharedDensityBuffer.Get(), static_cast<uint32_t>(regions.size()), regions.data());

        vkEndCommandBuffer(cmd);
        SubmitAndWaitForCommandBuffer(cmd);

        // m_DensityStagingBuffer is NOT destroyed here anymore - it's
        // persistent, reused by the next call. See
        // EnsureDensityStagingCapacity's own comment.
    }

    VoxelVolumeHandle VoxelSystemGPU::RegisterVolume(const VoxelVolumeDesc& desc, const std::vector<uint32_t>& chunkRemapOverride,
                                                      uint32_t compactChunkCountOverride)
    {
        // Reuse a freed slot before growing m_Volumes - see
        // VoxelVolumeHandle's comment and m_FreeSlotIndices' own
        // comment. Only throws capacity-exceeded when NOTHING is free
        // AND the high-water mark has hit kVoxelMaxVolumes - i.e. this
        // is now a concurrent-volume cap, not a lifetime one.
        VoxelVolumeIndex index;
        if (!m_FreeSlotIndices.empty())
        {
            index = m_FreeSlotIndices.back();
            m_FreeSlotIndices.pop_back();
            // Generation for this slot was already bumped when it was
            // freed (see UnregisterVolume) - m_Generations[index]
            // already holds the correct NEW generation to hand out.
        }
        else
        {
            if (m_Volumes.size() >= kVoxelMaxVolumes)
            {
                throw std::runtime_error(
                    "Eden: VoxelSystemGPU at capacity (see kVoxelMaxVolumes in VoxelTypes.h) - "
                    "kVoxelMaxVolumes volumes are concurrently registered and none are free to reuse. "
                    "Unregister some before registering more, or raise kVoxelMaxVolumes if this many "
                    "concurrent volumes is genuinely expected.");
            }
            index = static_cast<VoxelVolumeIndex>(m_Volumes.size());
            m_Volumes.emplace_back();
            m_Generations.push_back(0);
        }

        Volume volume;
        volume.desc = desc;

        uint32_t numChunks = desc.NumChunks();
        uint32_t numSamples = desc.NumSamples();

        // Zero-initialized here; SeedSphere is what actually gives this
        // meaningful values (uninitialized/all-zero density has no
        // defined inside/outside per kVoxelIsoLevel until then - don't
        // call MarchDirtyChunks before a Seed* call).
        volume.densityCPU.assign(numSamples, 0.0f);

        // Reuses a freed region if one fits, otherwise bump-allocates -
        // see AllocateDensityRegion. Already returns an offset aligned
        // to m_StorageBufferOffsetAlignmentBytes, legal as-is for the
        // VkDescriptorBufferInfo::offset bound below.
        volume.densityOffsetElements = AllocateDensityRegion(numSamples);

        // See RegisterVolume's own header comment for chunkRemapOverride/
        // compactChunkCountOverride - empty override means identity (every
        // existing caller before chunk compaction existed, and every
        // volume still subject to later Carve() edits, which a one-time
        // static classification can't safely handle).
        bool usingCompaction = !chunkRemapOverride.empty();
        if (usingCompaction && chunkRemapOverride.size() != numChunks)
        {
            throw std::runtime_error(
                "Eden: RegisterVolume's chunkRemapOverride has " + std::to_string(chunkRemapOverride.size())
                + " entries but desc describes " + std::to_string(numChunks)
                + " chunks - a mismatched remap table is a caller bug (see BuildChunkRemapTable, which "
                  "always sizes its output to match the SAME desc passed to it).");
        }
        volume.compactChunkCount = usingCompaction ? compactChunkCountOverride : numChunks;

        volume.vertices.Init(m_Allocator, static_cast<VkDeviceSize>(volume.compactChunkCount) * kVoxelMaxVerticesPerChunk * sizeof(Vertex),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        volume.indirect.Init(m_Allocator, static_cast<VkDeviceSize>(volume.compactChunkCount) * sizeof(IndirectCommand),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        volume.dirtyFlags.Init(m_Allocator, static_cast<VkDeviceSize>(numChunks) * sizeof(uint32_t),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // See ChunkRemapBuffer's comment in voxel_march.comp. Always
        // allocated (even for the identity/non-compacted case) so the
        // shader's addressing logic never needs to branch on whether a
        // given volume happens to be compacted - it always looks up
        // chunkRemap[chunkIndex], the identity case just makes that a
        // no-op indirection.
        volume.chunkRemap.Init(m_Allocator, static_cast<VkDeviceSize>(numChunks) * sizeof(uint32_t),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        if (usingCompaction)
        {
            UploadRange(volume.chunkRemap.Get(), 0, chunkRemapOverride.data(), chunkRemapOverride.size() * sizeof(uint32_t));
            // Mirror on the CPU - see Volume::chunkRemapCPU's comment for
            // why MarchDirtyChunks needs this rather than just re-deriving
            // it or reading the GPU buffer back.
            volume.chunkRemapCPU = chunkRemapOverride;
        }
        else
        {
            std::vector<uint32_t> identityRemap(numChunks);
            for (uint32_t i = 0; i < numChunks; ++i)
            {
                identityRemap[i] = i;
            }
            UploadRange(volume.chunkRemap.Get(), 0, identityRemap.data(), identityRemap.size() * sizeof(uint32_t));
            // Left empty (not identityRemap) for the non-compacted case -
            // Volume::chunkRemapCPU's comment documents empty as shorthand
            // for "chunkIndex is already the compact index", so every
            // MarchDirtyChunks call for the (much more common) non-compacted
            // case skips the lookup entirely instead of indexing into an
            // identity table that would just hand back the same value.
        }

        // Host-visible + persistently mapped: SetTransform is a plain
        // CPU memcpy, no staging round trip - identical reasoning to
        // ParticleSystemGPU's m_ColliderBuffers (small, infrequently
        // written, not worth a device-local + staging path).
        volume.instance.Init(m_Allocator, sizeof(InstanceData), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                              VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, &volume.instanceMapped);
        InstanceData defaultInstance{};
        defaultInstance.model = glm::mat4(1.0f);
        defaultInstance.colorOverride = glm::vec4(0.0f); // use the mesh's own (white) vertex color - see voxel_march.comp
        std::memcpy(volume.instanceMapped, &defaultInstance, sizeof(InstanceData));

        // Every chunk's slot in the shared vertex buffer is a fixed,
        // never-changing offset (compactIndex * kVoxelMaxVerticesPerChunk,
        // where compactIndex == the chunk's own raw index for the
        // identity/non-compacted case) - vertexCount starts at 0 until
        // the first MarchDirtyChunks call fills it in via atomic writes.
        // Only compactChunkCount entries now, not always numChunks - a
        // compacted volume's indirect buffer has no slot at all for a
        // chunk that was pre-classified as having no geometry, which is
        // exactly the point (see RegisterVolume's own header comment).
        std::vector<IndirectCommand> initialCommands(volume.compactChunkCount);
        for (uint32_t i = 0; i < volume.compactChunkCount; ++i)
        {
            initialCommands[i] = { 0, 1, i * kVoxelMaxVerticesPerChunk, 0 };
        }
        UploadRange(volume.indirect.Get(), 0, initialCommands.data(), initialCommands.size() * sizeof(IndirectCommand));

        // Reuse a recycled descriptor set (see m_FreeComputeSets'
        // comment) before allocating a new one from the pool - either
        // way, the vkUpdateDescriptorSets call below rewrites every
        // binding for THIS volume's buffers, so a reused set is
        // pointed at the right data before anything reads it.
        if (!m_FreeComputeSets.empty())
        {
            volume.computeSet = m_FreeComputeSets.back();
            m_FreeComputeSets.pop_back();
        }
        else
        {
            volume.computeSet = m_ComputeDescriptorPool.AllocateSet(m_ComputeSetLayout);
        }
        volume.chunkSolid.assign(numChunks, true);
        volume.dirtyChunkIndices.clear();

        std::array<VkDescriptorBufferInfo, 6> bufferInfos{};
        // Binding 0 is now the FULL shared density buffer (offset 0,
        // VK_WHOLE_SIZE), not a per-volume sub-range like it used to be.
        // Ghost-sample neighbor reads (voxel_march.comp's DensityAt)
        // need a volume's compute set to be able to reach into an
        // ADJACENT terrain tile's region of this same buffer for
        // gradient continuity across tile boundaries - a sub-range view
        // can't do that, Vulkan won't let a shader read outside a
        // binding's declared range. Every density read now goes through
        // an explicit element offset from push constants
        // (ownDensityElementOffset for this volume's own samples, or
        // one of the 4 neighborDensityOffset* fields when a boundary
        // cell's gradient reaches into a neighbor) instead of relying on
        // the binding's offset to do it implicitly - see
        // VoxelParamsGPU's own comment for why.
        bufferInfos[0] = { m_SharedDensityBuffer.Get(), 0, VK_WHOLE_SIZE };
        bufferInfos[1] = { volume.vertices.Get(), 0, VK_WHOLE_SIZE };
        bufferInfos[2] = { volume.indirect.Get(), 0, VK_WHOLE_SIZE };
        bufferInfos[3] = { m_MC33Table.Get(), 0, VK_WHOLE_SIZE };
        bufferInfos[4] = { volume.dirtyFlags.Get(), 0, VK_WHOLE_SIZE };
        bufferInfos[5] = { volume.chunkRemap.Get(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 6> writes{};
        for (uint32_t b = 0; b < writes.size(); ++b)
        {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = volume.computeSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bufferInfos[b];
        }
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        m_Volumes[index] = std::move(volume);
        return MakeVoxelVolumeHandle(index, m_Generations[index]);
    }

    VoxelSystemGPU::Volume& VoxelSystemGPU::GetVolume(VoxelVolumeHandle handle)
    {
        if (!IsValid(handle))
        {
            throw std::runtime_error(
                "Eden: VoxelVolumeHandle (index " + std::to_string(GetVoxelVolumeIndex(handle))
                + ", generation " + std::to_string(GetVoxelVolumeGeneration(handle))
                + ") is stale - its slot was freed by UnregisterVolume and may have been reused by "
                  "a different volume since. A caller held onto this handle past its volume's "
                  "lifetime.");
        }
        return m_Volumes[GetVoxelVolumeIndex(handle)];
    }

    const VoxelSystemGPU::Volume& VoxelSystemGPU::GetVolume(VoxelVolumeHandle handle) const
    {
        if (!IsValid(handle))
        {
            throw std::runtime_error(
                "Eden: VoxelVolumeHandle (index " + std::to_string(GetVoxelVolumeIndex(handle))
                + ", generation " + std::to_string(GetVoxelVolumeGeneration(handle))
                + ") is stale - its slot was freed by UnregisterVolume and may have been reused by "
                  "a different volume since. A caller held onto this handle past its volume's "
                  "lifetime.");
        }
        return m_Volumes[GetVoxelVolumeIndex(handle)];
    }

    bool VoxelSystemGPU::IsValid(VoxelVolumeHandle handle) const
    {
        VoxelVolumeIndex index = GetVoxelVolumeIndex(handle);
        return index < m_Volumes.size() && m_Generations[index] == GetVoxelVolumeGeneration(handle);
    }

    VkDeviceSize VoxelSystemGPU::AllocateDensityRegion(VkDeviceSize numSamples)
    {
        VkDeviceSize alignmentElements =
            std::max<VkDeviceSize>(1, m_StorageBufferOffsetAlignmentBytes / sizeof(float));

        // First-fit: first free region (in whatever order they sit in
        // the vector - coalescing keeps it offset-sorted, see
        // ReleaseDensityRegion) whose ALIGNED usable span is big enough.
        for (size_t i = 0; i < m_FreeDensityRegions.size(); ++i)
        {
            FreeDensityRegion& region = m_FreeDensityRegions[i];
            VkDeviceSize alignedStart =
                ((region.offsetElements + alignmentElements - 1) / alignmentElements) * alignmentElements;
            VkDeviceSize alignmentWaste = alignedStart - region.offsetElements;
            if (alignmentWaste >= region.sizeElements)
            {
                continue; // alignment padding alone consumes the whole region
            }
            VkDeviceSize usableSize = region.sizeElements - alignmentWaste;
            if (usableSize < numSamples)
            {
                continue;
            }

            VkDeviceSize allocatedEnd = alignedStart + numSamples;
            VkDeviceSize regionEnd = region.offsetElements + region.sizeElements;

            if (allocatedEnd >= regionEnd)
            {
                // Consumes the rest of the region (modulo any small
                // unaligned gap in front, which is lost - not worth
                // tracking a separate sub-alignment-sized free region
                // for a handful of bytes).
                m_FreeDensityRegions.erase(m_FreeDensityRegions.begin() + i);
            }
            else
            {
                // Shrink this region down to the unused tail past what
                // was just handed out.
                region.offsetElements = allocatedEnd;
                region.sizeElements = regionEnd - allocatedEnd;
            }
            return alignedStart;
        }

        // Nothing free fits - fall back to the original bump-allocate
        // path (grow past every region ever handed out).
        VkDeviceSize alignedOffset =
            ((m_SharedDensityBufferUsedElements + alignmentElements - 1) / alignmentElements) * alignmentElements;

        if (alignedOffset + numSamples > m_SharedDensityBufferCapacityElements)
        {
            throw std::runtime_error(
                "Eden: VoxelSystemGPU's shared density buffer is out of space (requested "
                + std::to_string(numSamples) + " more elements at offset " + std::to_string(alignedOffset)
                + ", capacity is " + std::to_string(m_SharedDensityBufferCapacityElements)
                + ") - raise kSharedDensityBufferCapacityElements in VoxelTypes.h, or this volume's "
                  "sample count/resolution is larger than this project's current budget expects.");
        }

        m_SharedDensityBufferUsedElements = alignedOffset + numSamples;
        return alignedOffset;
    }

    void VoxelSystemGPU::ReleaseDensityRegion(VkDeviceSize offsetElements, VkDeviceSize numSamples)
    {
        if (numSamples == 0)
        {
            return;
        }

        m_FreeDensityRegions.push_back({ offsetElements, numSamples });

        // Coalesce: sort by offset, then merge any run of regions where
        // one's end exactly equals the next one's start. O(n log n) in
        // the free-region count, which is bounded by kVoxelMaxVolumes
        // (64) - not worth a fancier structure at this scale.
        std::sort(m_FreeDensityRegions.begin(), m_FreeDensityRegions.end(),
                  [](const FreeDensityRegion& a, const FreeDensityRegion& b) { return a.offsetElements < b.offsetElements; });

        std::vector<FreeDensityRegion> merged;
        merged.reserve(m_FreeDensityRegions.size());
        for (const FreeDensityRegion& region : m_FreeDensityRegions)
        {
            if (!merged.empty() && merged.back().offsetElements + merged.back().sizeElements == region.offsetElements)
            {
                merged.back().sizeElements += region.sizeElements;
            }
            else
            {
                merged.push_back(region);
            }
        }
        m_FreeDensityRegions = std::move(merged);
    }

    void VoxelSystemGPU::UnregisterVolume(VoxelVolumeHandle handle)
    {
        // Not routed through GetVolume() - calling UnregisterVolume
        // twice on the same handle, or on a handle whose slot has
        // since been recycled into an unrelated volume, must be a safe
        // no-op (matches Registry::DestroyEntity's own "not optional"
        // double-destroy guard - without it, this could free a slot
        // that's already been reused out from under a live volume).
        if (!IsValid(handle))
        {
            return;
        }

        // CRITICAL - this is not optional, and was the direct cause of
        // a real crash (VK_ERROR_OUT_OF_DEVICE_MEMORY / lost device /
        // VMA "allocations not freed" assertion on shutdown - a full
        // command-buffer-in-flight violation, not a leak or a cosmetic
        // validation warning).
        //
        // Everything below - destroying this volume's buffers, and
        // recycling its descriptor set into m_FreeComputeSets for the
        // NEXT RegisterVolume to immediately vkUpdateDescriptorSets
        // over - assumes nothing on the GPU is still touching them.
        // That's false in general: a march/render/collision dispatch
        // recorded against this volume earlier THIS FRAME (or a
        // previous frame, if anything is still in flight) may not have
        // finished executing yet when UnregisterVolume runs on the CPU
        // timeline. Destroying a buffer, or rewriting a descriptor
        // set's bindings, while a command buffer that references it is
        // still pending is a Vulkan validation error family
        // (vkDestroyBuffer/vkDestroyPipeline "currently in use by
        // VkCommandBuffer", vkUpdateDescriptorSets "... is in use by
        // VkCommandBuffer ... only possible with VK_EXT_descriptor_
        // indexing") - MoltenVK doesn't merely warn about this, it can
        // silently hand the reused buffer/descriptor a stale in-flight
        // command's writes, corrupting VMA's own bookkeeping and
        // losing the whole VkDevice.
        //
        // A full stall is the correct fix here, not a workaround around
        // it - UnregisterVolume is a rare, deliberate action (editor
        // "Destroy" click, or a future melt/reform-driven cleanup - see
        // this function's own header comment), never a per-frame hot
        // path, so paying for full GPU idle here is the right trade
        // against the alternative (a proper per-resource fence/frames-
        // in-flight tracker), which is real, more scoped work this
        // project doesn't have the infrastructure for yet - flagged
        // explicitly rather than half-built. Same
        // "vkDeviceWaitIdle before destroying" pattern already used at
        // shutdown (VulkanDevice.cpp), just invoked mid-session instead
        // of only at teardown.
        vkDeviceWaitIdle(m_Device);

        VoxelVolumeIndex index = GetVoxelVolumeIndex(handle);
        Volume& volume = m_Volumes[index];

        // Frees each buffer's VMA allocation immediately (Shutdown(),
        // not waiting for the destructor) - see VulkanBuffer::Shutdown.
        volume.vertices.Shutdown();
        volume.indirect.Shutdown();
        volume.dirtyFlags.Shutdown();
        volume.chunkRemap.Shutdown();
        volume.instance.Shutdown();
        volume.instanceMapped = nullptr; // the memory it pointed into was just freed above

        // Reclaims CPU memory too, not just marks it "empty" - swap-with-
        // empty rather than .clear() so capacity is actually released
        // (matches how large densityCPU can get at higher resolutions;
        // .clear() alone would keep the allocation reserved).
        volume.hasPendingUpload = false;
        std::vector<float>().swap(volume.densityCPU);
        std::vector<uint32_t>().swap(volume.dirtyChunkIndices);
        std::vector<bool>().swap(volume.chunkSolid);

        // Real reclaim, not a leak - see AllocateDensityRegion/
        // ReleaseDensityRegion and m_FreeComputeSets' own comments.
        ReleaseDensityRegion(volume.densityOffsetElements, volume.desc.NumSamples());
        m_FreeComputeSets.push_back(volume.computeSet);
        volume.computeSet = VK_NULL_HANDLE;

        // Recycle the slot itself - bump generation FIRST so any handle
        // built from the old generation now fails IsValid, THEN release
        // the index for reuse (mirrors Registry::DestroyEntity exactly).
        m_Generations[index]++;
        m_FreeSlotIndices.push_back(index);
    }

    VkBuffer VoxelSystemGPU::GetVertexBuffer(VoxelVolumeHandle handle) const { return GetVolume(handle).vertices.Get(); }
    VkBuffer VoxelSystemGPU::GetIndirectBuffer(VoxelVolumeHandle handle) const { return GetVolume(handle).indirect.Get(); }
    VkBuffer VoxelSystemGPU::GetInstanceBuffer(VoxelVolumeHandle handle) const { return GetVolume(handle).instance.Get(); }
    uint32_t VoxelSystemGPU::GetChunkCount(VoxelVolumeHandle handle) const { return GetVolume(handle).compactChunkCount; }
    VkBuffer VoxelSystemGPU::GetDensityBuffer(VoxelVolumeHandle /*handle*/) const { return m_SharedDensityBuffer.Get(); }
    VkDeviceSize VoxelSystemGPU::GetDensityBufferOffset(VoxelVolumeHandle handle) const { return GetVolume(handle).densityOffsetElements; }
    glm::ivec3 VoxelSystemGPU::GetSampleDims(VoxelVolumeHandle handle) const { return GetVolume(handle).desc.SampleDims(); }
    float VoxelSystemGPU::GetVoxelSize(VoxelVolumeHandle handle) const { return GetVolume(handle).desc.voxelSize; }

    void VoxelSystemGPU::SetTransform(VoxelVolumeHandle handle, const glm::mat4& model, const glm::vec4& colorOverride)
    {
        Volume& volume = GetVolume(handle);
        volume.lastColor = colorOverride; // see GetVolumeColor's comment - mirrored for later readback, doesn't change this function's own behavior
        InstanceData instance{};
        instance.model = model;
        instance.colorOverride = colorOverride;
        // Direct write through the mapped pointer captured at Init time
        // (RegisterVolume) - no GPU command buffer involved. This used
        // to route through UploadRange (a staged vkCmdCopyBuffer)
        // because volume.instanceMapped didn't exist yet and this had
        // no other way to reach the mapped memory - but `instance` was
        // never given VK_BUFFER_USAGE_TRANSFER_DST_BIT (there was never
        // a reason to; it's host-visible and mapped specifically so it
        // WOULDN'T need copying into), making that old call invalid
        // Vulkan on every single invocation: "dstBuffer was created
        // with VERTEX_BUFFER_BIT but requires TRANSFER_DST_BIT". That
        // one invalid vkCmdCopyBuffer call was corrupting GPU/validation
        // state from the very first SetTransform of a session - not
        // something introduced by volume register/unregister reuse, and
        // not fixed by anything in UnregisterVolume's synchronization.
        std::memcpy(volume.instanceMapped, &instance, sizeof(InstanceData));
    }

    VkDeviceSize VoxelSystemGPU::EstimateVertexBufferBytes(const VoxelVolumeDesc& desc, uint32_t compactChunkCountOverride)
    {
        // Byte-for-byte the same formula RegisterVolume's real
        // vertices.Init call uses just above - see that line and
        // EstimateVertexBufferBytes' own header comment. 0 (default)
        // matches RegisterVolume's own "0 means identity/full" convention.
        uint32_t chunkCount = compactChunkCountOverride == 0 ? desc.NumChunks() : compactChunkCountOverride;
        return static_cast<VkDeviceSize>(chunkCount) * kVoxelMaxVerticesPerChunk * sizeof(Vertex);
    }

    void VoxelSystemGPU::SetChemicalId(VoxelVolumeHandle handle, uint32_t chemicalId)
    {
        GetVolume(handle).chemicalId = chemicalId;
    }

    uint32_t VoxelSystemGPU::GetChemicalId(VoxelVolumeHandle handle) const
    {
        return GetVolume(handle).chemicalId;
    }

    glm::vec4 VoxelSystemGPU::GetVolumeColor(VoxelVolumeHandle handle) const
    {
        return GetVolume(handle).lastColor;
    }

    void VoxelSystemGPU::SeedSphere(VoxelVolumeHandle handle, const glm::vec3& localCenter, float radius)
    {
        Volume& volume = GetVolume(handle);
        glm::ivec3 sampleDims = volume.desc.SampleDims();

        // Plain CPU triple loop, same math voxel_seed.comp used to do -
        // see the class comment in VoxelSystemGPU.h on why this moved
        // off the GPU. Trivial cost at this volume's sample counts.
        for (int32_t z = 0; z < sampleDims.z; ++z)
        for (int32_t y = 0; y < sampleDims.y; ++y)
        for (int32_t x = 0; x < sampleDims.x; ++x)
        {
            glm::vec3 localPos = glm::vec3(x, y, z) * volume.desc.voxelSize;
            float d = glm::length(localPos - localCenter) - radius;
            volume.densityCPU[SampleIndex(volume.desc, glm::ivec3(x, y, z))] = d;
        }

        UploadDensity(volume);

        // Every chunk needs re-marching after a full-field seed - the
        // actual GPU dirtyFlags buffer write for this happens in
        // MarchDirtyChunks (keyed off this CPU list), not here - see
        // Volume::dirtyChunkIndices' comment in the header for why that
        // single source of truth is shared with Carve().
        volume.dirtyChunkIndices.resize(volume.desc.NumChunks());
        for (uint32_t i = 0; i < volume.desc.NumChunks(); ++i)
        {
            volume.dirtyChunkIndices[i] = i;
        }
    }

    void VoxelSystemGPU::SeedBox(VoxelVolumeHandle handle, const glm::vec3& localCenter, const glm::vec3& halfExtents)
    {
        Volume& volume = GetVolume(handle);
        glm::ivec3 sampleDims = volume.desc.SampleDims();

        // Identical shape to SeedSphere's loop, SDF::Box instead of the
        // inline sphere formula - reused directly rather than
        // reimplemented so a voxel-box's surface matches
        // ColliderShape::Box's analytic surface exactly.
        for (int32_t z = 0; z < sampleDims.z; ++z)
        for (int32_t y = 0; y < sampleDims.y; ++y)
        for (int32_t x = 0; x < sampleDims.x; ++x)
        {
            glm::vec3 localPos = glm::vec3(x, y, z) * volume.desc.voxelSize;
            float d = SDF::Box(localPos - localCenter, halfExtents);
            volume.densityCPU[SampleIndex(volume.desc, glm::ivec3(x, y, z))] = d;
        }

        UploadDensity(volume);

        volume.dirtyChunkIndices.resize(volume.desc.NumChunks());
        for (uint32_t i = 0; i < volume.desc.NumChunks(); ++i)
        {
            volume.dirtyChunkIndices[i] = i;
        }
    }

    void VoxelSystemGPU::SplatParticlesIntoDensityField(std::vector<float>& densityCPU, const VoxelVolumeDesc& desc,
                                                         const std::vector<glm::vec3>& localPositions, float particleRadius,
                                                         float smoothRadius)
    {
        glm::ivec3 sampleDims = desc.SampleDims();
        float voxelSize = desc.voxelSize;

        // "Splat" approach (every PARTICLE only touches the small
        // neighborhood of samples it can actually influence, rather than
        // every SAMPLE searching outward for nearby particles) - see
        // SeedFromParticles' own comment (this function used to BE its
        // body directly) for the full performance story behind this
        // choice. Extracted into a standalone, Volume-free function so
        // ReformSystem can compute a cluster's real density field BEFORE
        // ever registering a volume for it - see CountSurfaceCrossingChunks
        // for what that classification is actually used for.
        std::fill(densityCPU.begin(), densityCPU.end(), kVoxelIsoLevel + 1.0f);

        float influenceRadius = particleRadius + smoothRadius;

        for (const glm::vec3& particlePos : localPositions)
        {
            glm::ivec3 sampleMin = glm::ivec3(glm::floor((particlePos - glm::vec3(influenceRadius)) / voxelSize));
            glm::ivec3 sampleMax = glm::ivec3(glm::ceil((particlePos + glm::vec3(influenceRadius)) / voxelSize));
            sampleMin = glm::max(sampleMin, glm::ivec3(0));
            sampleMax = glm::min(sampleMax, sampleDims - glm::ivec3(1));

            for (int32_t z = sampleMin.z; z <= sampleMax.z; ++z)
            for (int32_t y = sampleMin.y; y <= sampleMax.y; ++y)
            for (int32_t x = sampleMin.x; x <= sampleMax.x; ++x)
            {
                glm::vec3 localPos = glm::vec3(x, y, z) * voxelSize;

                // Same smooth union of per-particle spheres via SmoothMin
                // as before - still an order-dependent approximation of a
                // true N-way blend (particles are folded in whatever
                // order localPositions happens to list them, not
                // distance-sorted), same as this algorithm has always
                // been - no visual quality change from this extraction,
                // only WHERE the code lives changed.
                float d = glm::length(localPos - particlePos) - particleRadius;
                size_t sampleIndex = SampleIndex(desc, glm::ivec3(x, y, z));
                densityCPU[sampleIndex] = SmoothMin(densityCPU[sampleIndex], d, smoothRadius);
            }
        }
    }

    uint32_t VoxelSystemGPU::CountSurfaceCrossingChunks(const std::vector<float>& densityCPU, const VoxelVolumeDesc& desc)
    {
        // A chunk contributes triangles if and only if the marching-cubes
        // algorithm finds at least one solid sample and at least one
        // empty sample among the samples it owns - see voxel_march.comp's
        // own per-cell logic for the (equivalent, finer-grained) real
        // test this approximates at chunk granularity. A chunk entirely
        // solid or entirely empty never emits a single triangle,
        // regardless of resolution - this is what a uniform "reserve
        // worst-case space for every chunk" vertex buffer sizing wastes
        // space on. See BuildChunkRemapTable, which does this SAME
        // classification but also builds the actual GPU-side remap table
        // that makes shrinking the buffer to fit ONLY these chunks safe -
        // this function alone was the first, non-load-bearing half of
        // that (a count, not something the shader can act on by itself).
        glm::ivec3 sampleDims = desc.SampleDims();
        glm::ivec3 chunkDims = desc.chunkDims;
        int32_t chunkSize = static_cast<int32_t>(kVoxelChunkSize);

        uint32_t crossingChunks = 0;

        for (int32_t cz = 0; cz < chunkDims.z; ++cz)
        for (int32_t cy = 0; cy < chunkDims.y; ++cy)
        for (int32_t cx = 0; cx < chunkDims.x; ++cx)
        {
            glm::ivec3 sampleMin = glm::ivec3(cx, cy, cz) * chunkSize;
            glm::ivec3 sampleMax = glm::min(sampleMin + glm::ivec3(chunkSize), sampleDims - glm::ivec3(1));

            bool hasSolid = false;
            bool hasEmpty = false;

            for (int32_t z = sampleMin.z; z <= sampleMax.z && !(hasSolid && hasEmpty); ++z)
            for (int32_t y = sampleMin.y; y <= sampleMax.y && !(hasSolid && hasEmpty); ++y)
            for (int32_t x = sampleMin.x; x <= sampleMax.x && !(hasSolid && hasEmpty); ++x)
            {
                float d = densityCPU[SampleIndex(desc, glm::ivec3(x, y, z))];
                if (d < kVoxelIsoLevel)
                {
                    hasSolid = true;
                }
                else
                {
                    hasEmpty = true;
                }
            }

            if (hasSolid && hasEmpty)
            {
                ++crossingChunks;
            }
        }

        return crossingChunks;
    }

    uint32_t VoxelSystemGPU::BuildChunkRemapTable(const std::vector<float>& densityCPU, const VoxelVolumeDesc& desc,
                                                   std::vector<uint32_t>& outRemap)
    {
        // Same classification as CountSurfaceCrossingChunks (see that
        // function's comment for what "crosses the isosurface" means and
        // why it's the right test) - this version additionally assigns
        // each crossing chunk a compact index as it goes (0, 1, 2... in
        // raw-chunk-index order) and records kVoxelChunkRemapUnused for
        // every other chunk, instead of just counting. See
        // voxel_march.comp's ChunkRemapBuffer for how the shader consumes
        // this, and RegisterVolume's chunkRemapOverride for how a caller
        // installs it.
        glm::ivec3 sampleDims = desc.SampleDims();
        glm::ivec3 chunkDims = desc.chunkDims;
        int32_t chunkSize = static_cast<int32_t>(kVoxelChunkSize);
        uint32_t numChunks = desc.NumChunks();

        outRemap.assign(numChunks, kVoxelChunkRemapUnused);
        uint32_t compactCount = 0;

        for (int32_t cz = 0; cz < chunkDims.z; ++cz)
        for (int32_t cy = 0; cy < chunkDims.y; ++cy)
        for (int32_t cx = 0; cx < chunkDims.x; ++cx)
        {
            glm::ivec3 sampleMin = glm::ivec3(cx, cy, cz) * chunkSize;
            glm::ivec3 sampleMax = glm::min(sampleMin + glm::ivec3(chunkSize), sampleDims - glm::ivec3(1));

            bool hasSolid = false;
            bool hasEmpty = false;

            for (int32_t z = sampleMin.z; z <= sampleMax.z && !(hasSolid && hasEmpty); ++z)
            for (int32_t y = sampleMin.y; y <= sampleMax.y && !(hasSolid && hasEmpty); ++y)
            for (int32_t x = sampleMin.x; x <= sampleMax.x && !(hasSolid && hasEmpty); ++x)
            {
                float d = densityCPU[SampleIndex(desc, glm::ivec3(x, y, z))];
                if (d < kVoxelIsoLevel)
                {
                    hasSolid = true;
                }
                else
                {
                    hasEmpty = true;
                }
            }

            if (hasSolid && hasEmpty)
            {
                // Raw index formula must match voxel_march.comp's own
                // chunkIndex computation exactly (chunkCoord.x +
                // chunkCoord.y*chunkDims.x + chunkCoord.z*chunkDims.x*chunkDims.y)
                // - this is the SAME flat-index convention SampleIndex
                // and every other chunk-indexed buffer in this class
                // already use, not a new one invented here.
                int32_t rawChunkIndex = cx + cy * chunkDims.x + cz * chunkDims.x * chunkDims.y;
                outRemap[rawChunkIndex] = compactCount;
                ++compactCount;
            }
        }

        return compactCount;
    }

    void VoxelSystemGPU::MarkAllChunksDirty(Volume& volume)
    {
        uint32_t numChunks = volume.desc.NumChunks();
        volume.dirtyChunkIndices.resize(numChunks);
        for (uint32_t i = 0; i < numChunks; ++i)
        {
            volume.dirtyChunkIndices[i] = i;
        }
    }

    void VoxelSystemGPU::SetDensityFieldDirect(VoxelVolumeHandle handle, std::vector<float> densityCPU)
    {
        Volume& volume = GetVolume(handle);
        volume.densityCPU = std::move(densityCPU);
        UploadDensity(volume);
        MarkAllChunksDirty(volume);
    }

    void VoxelSystemGPU::SeedFromParticles(VoxelVolumeHandle handle, const std::vector<glm::vec3>& localPositions, float particleRadius,
                                            float smoothRadius)
    {
        Volume& volume = GetVolume(handle);

        // Rewritten from a "gather" approach (every SAMPLE searches
        // outward through a spatial hash for nearby particles) to a
        // "splat" approach (every PARTICLE only touches the small
        // neighborhood of samples it can actually influence). The old
        // version's cost scaled with total sample count regardless of
        // how many particles actually existed - for a real measured
        // case (724 particles, a wide/thin puddle needing a
        // 20x3x16-chunk volume), that meant iterating ~500K+ samples,
        // most sitting in genuinely empty space, each paying for an
        // escalating grid search (up to 729 hash lookups across 4
        // radii) before concluding "nothing here" - 13.6+ SECONDS,
        // confirmed via direct timing, and a coarse AABB-reject attempt
        // at trimming that search didn't meaningfully help (the reject
        // margin needed to stay correct ended up nearly as wide as the
        // volume itself in the puddle's thin axis, canceling out most of
        // the intended savings).
        //
        // This version's cost scales with PARTICLE count instead, and is
        // now implemented in SplatParticlesIntoDensityField (see that
        // function) so ReformSystem can also call it standalone, before a
        // volume even exists, purely to classify which chunks will
        // actually need geometry - see CountSurfaceCrossingChunks.
        SplatParticlesIntoDensityField(volume.densityCPU, volume.desc, localPositions, particleRadius, smoothRadius);
        UploadDensity(volume);
        MarkAllChunksDirty(volume);
    }

    void VoxelSystemGPU::SeedHeightfieldNoise(VoxelVolumeHandle handle, float baseHeight, float amplitude,
                                               float frequency, int octaves, uint32_t seed)
    {
        Volume& volume = GetVolume(handle);
        glm::ivec3 sampleDims = volume.desc.SampleDims();

        // Arbitrary irrational multipliers so nearby integer seeds don't
        // just look like a panned copy of the same noise field - not
        // meant to be a real hash, just enough decorrelation that
        // seed=1 and seed=2 read as genuinely different terrain.
        glm::vec2 seedOffset(static_cast<float>(seed) * 6180.339887f, static_cast<float>(seed) * 3141.592653f);

        for (int32_t z = 0; z < sampleDims.z; ++z)
        for (int32_t y = 0; y < sampleDims.y; ++y)
        for (int32_t x = 0; x < sampleDims.x; ++x)
        {
            glm::vec3 localPos = glm::vec3(x, y, z) * volume.desc.voxelSize;
            glm::vec3 worldPos = volume.desc.origin + localPos;

            // Standard fBm: each octave halves amplitude and doubles
            // frequency of the previous one. Only x/z feed the noise -
            // this is a 2D heightfield, not 3D noise (no overhangs/caves
            // yet - see this function's header comment on scope).
            float height = 0.0f;
            float amp = amplitude;
            float freq = frequency;
            for (int o = 0; o < octaves; ++o)
            {
                height += glm::perlin(glm::vec2(worldPos.x, worldPos.z) * freq + seedOffset) * amp;
                amp *= 0.5f;
                freq *= 2.0f;
            }

            float terrainHeight = baseHeight + height;

            // Detail layer, separate from the macro fBm loop above and
            // deliberately NOT just "one more octave" of it - doubling
            // frequency/halving amplitude each octave decays amplitude
            // far faster than frequency climbs into a range that
            // actually reads as surface texture at normal viewing
            // distance (reaching a ~2m period this way would need ~7
            // more octaves, by which point amplitude is under a
            // millimeter - invisible). This is a single fixed high-
            // frequency, low-amplitude term instead, scaled off the
            // caller's own amplitude (so amplitude=0 disables it too,
            // and it stays sensible if the macro amplitude changes)
            // purely to give the surface enough per-vertex normal
            // variation for Blinn-Phong shading to show something on an
            // otherwise near-flat field - NOT meant to be geometrically
            // significant enough to affect carving or particle rest
            // behavior. Different seed multiplier than the macro
            // offset so the detail ripple doesn't visibly correlate
            // with the macro shape.
            float detailHeight = glm::perlin(glm::vec2(worldPos.x, worldPos.z) * (frequency * 12.0f) + seedOffset * 1.7f)
                                  * (amplitude * 0.08f);
            terrainHeight += detailHeight;

            // Vertical-offset pseudo-SDF (see header comment for why
            // this isn't an exact distance off-flat) - negative below
            // the surface (solid), positive above (empty), matching
            // every other Seed*'s inside-is-negative convention.
            float d = worldPos.y - terrainHeight;
            volume.densityCPU[SampleIndex(volume.desc, glm::ivec3(x, y, z))] = d;
        }

        UploadDensity(volume);

        volume.dirtyChunkIndices.resize(volume.desc.NumChunks());
        for (uint32_t i = 0; i < volume.desc.NumChunks(); ++i)
        {
            volume.dirtyChunkIndices[i] = i;
        }
    }

    void VoxelSystemGPU::SubmitAndWaitForCommandBuffer(VkCommandBuffer cmd)
    {
        // A fence signals when THIS submission's work completes, unlike
        // vkQueueWaitIdle which blocks until the ENTIRE queue drains -
        // see this function's own header comment (VoxelSystemGPU.h) for
        // why that difference is the actual point here, not just a style
        // preference: MarchDirtyChunks and UploadDensityRegion both run
        // on every single Carve(), and with carving now holdable,
        // vkQueueWaitIdle there meant every carve tick could stall behind
        // an unrelated in-flight frame already queued on this same
        // queue - a real, measured cause of stutter while holding the
        // carve button, not a theoretical one.
        //
        // Reuses m_SubmitFence (created once in Init, reset here every
        // call) rather than creating and destroying a fresh VkFence per
        // call, which is what this function did at first. That turned
        // out to be a real regression, not a free win: VkFence
        // create/destroy is real driver work, not just a CPU-side struct,
        // and doing it twice per carve tick (once here, once from
        // UploadDensityRegion's own call) on MoltenVK specifically - this
        // project's primary target, and already flagged in prior
        // learnings as having its own sync/descriptor quirks separate
        // from desktop Vulkan - plausibly cost more than the
        // vkQueueWaitIdle stall it was replacing. Resetting and reusing
        // one persistent fence keeps the actual fix (don't wait for
        // unrelated in-flight work) without paying object-churn cost on
        // every single carve tick.
        //
        // cmd itself is never freed here anymore either - it's always
        // m_ReusableCommandBuffer (see GetReusableCommandBuffer), the
        // SAME class of per-call-churn mistake as the fence, on the
        // command buffer this time - missed in the two previous passes
        // that fixed the fence and the staging buffer in these exact
        // same functions.
        if (m_SubmitFence == VK_NULL_HANDLE)
        {
            // Fallback matches Init's own comment: no working fence
            // available, so fall back to the always-correct-if-slower
            // vkQueueWaitIdle path rather than proceeding with no wait
            // at all - a caller reading GPU results right after this
            // call still needs a real guarantee the write landed.
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_GraphicsQueue);
            return;
        }

        // Must reset before reuse - a signaled fence's next
        // vkWaitForFences call would return immediately on the OLD
        // signal rather than waiting for THIS submission.
        vkResetFences(m_Device, 1, &m_SubmitFence);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, m_SubmitFence);

        // No timeout - matches vkQueueWaitIdle's own "wait as long as it
        // takes" contract exactly; this is a correctness wait, not a
        // polling one.
        vkWaitForFences(m_Device, 1, &m_SubmitFence, VK_TRUE, UINT64_MAX);
    }

    void VoxelSystemGPU::MarchDirtyChunks(VoxelVolumeHandle handle)
    {
        EDEN_PROFILE_SCOPE("Voxel: march dirty chunks (total, incl. upload)");
        FlushPendingUpload(handle); // the march reads the GPU density buffer
        Volume& volume = GetVolume(handle);

        if (volume.dirtyChunkIndices.empty())
        {
            return; // nothing to do - e.g. called after a Carve() that missed the volume entirely
        }

        VoxelParamsGPU params{};
        params.origin = volume.desc.origin;
        params.voxelSize = volume.desc.voxelSize;
        params.sampleDims = volume.desc.SampleDims();
        params.chunkVoxelSize = static_cast<int32_t>(kVoxelChunkSize);
        params.chunkDims = volume.desc.chunkDims;
        params.maxVerticesPerChunk = static_cast<int32_t>(kVoxelMaxVerticesPerChunk);
        params.isoLevel = kVoxelIsoLevel;

        // Ghost-sample offsets - see VoxelParamsGPU's own comment and
        // voxel_march.comp's DensityAt. ownDensityElementOffset always
        // needs setting now (binding 0 is the full shared buffer, not a
        // per-volume sub-range - see RegisterVolume); the 4 neighbor
        // offsets stay -1 (their default) for any direction
        // SetVolumeNeighbors was never called for, or was called with
        // InvalidVoxelVolumeHandle for - which is every non-terrain
        // volume and every terrain tile on the grid's outer edge.
        params.ownDensityElementOffset = static_cast<int32_t>(volume.densityOffsetElements);
        if (IsValid(volume.neighborNegX)) params.neighborDensityOffsetNegX = static_cast<int32_t>(GetVolume(volume.neighborNegX).densityOffsetElements);
        if (IsValid(volume.neighborPosX)) params.neighborDensityOffsetPosX = static_cast<int32_t>(GetVolume(volume.neighborPosX).densityOffsetElements);
        if (IsValid(volume.neighborNegZ)) params.neighborDensityOffsetNegZ = static_cast<int32_t>(GetVolume(volume.neighborNegZ).densityOffsetElements);
        if (IsValid(volume.neighborPosZ)) params.neighborDensityOffsetPosZ = static_cast<int32_t>(GetVolume(volume.neighborPosZ).densityOffsetElements);

        VkCommandBuffer cmd = GetReusableCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Set the GPU dirtyFlags entry (so the shader's per-cell gate
        // lets these chunks' cells through) and reset vertexCount to 0
        // (so this march's atomicAdds start from a clean count instead
        // of accumulating on top of a previous march's leftover total -
        // matters for Carve()-triggered re-marches of an
        // already-marched chunk, not just the first-ever march) for
        // ONLY the chunks in volume.dirtyChunkIndices - NOT the whole
        // buffer. A Carve() touching 2 chunks out of 8 must not zero the
        // other 6 chunks' vertexCount, or they'd vanish from the draw
        // (their own geometry is still valid, they just weren't
        // re-marched this call).
        for (uint32_t chunkIndex : volume.dirtyChunkIndices)
        {
            // dirtyFlags is always sized to the RAW chunk count (see its
            // own comment on Volume), so this one is never compacted -
            // raw chunkIndex is always the right address here.
            vkCmdFillBuffer(cmd, volume.dirtyFlags.Get(), static_cast<VkDeviceSize>(chunkIndex) * sizeof(uint32_t), sizeof(uint32_t), 1);

            // `indirect` IS compacted (compactChunkCount entries, not
            // numChunks) - see Volume::chunkRemapCPU's comment. Translate
            // through the same remap the GPU shader uses, and skip chunks
            // with no compact slot at all (kVoxelChunkRemapUnused) instead
            // of resetting a slot that doesn't exist - this is what a
            // caller marking every raw chunk dirty on a compacted volume
            // (ClearVolume, SetDensityFieldDirect, a Seed*) needs to stay
            // in-bounds.
            uint32_t compactIndex = volume.chunkRemapCPU.empty() ? chunkIndex : volume.chunkRemapCPU[chunkIndex];
            if (compactIndex == kVoxelChunkRemapUnused)
            {
                continue;
            }
            VkDeviceSize vertexCountOffset = static_cast<VkDeviceSize>(compactIndex) * sizeof(IndirectCommand);
            vkCmdFillBuffer(cmd, volume.indirect.Get(), vertexCountOffset, sizeof(uint32_t), 0);
        }

        VkMemoryBarrier toCompute{};
        toCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 1, &toCompute, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_MarchPipeline.Get());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComputePipelineLayout.Get(),
                                 0, 1, &volume.computeSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_ComputePipelineLayout.Get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VoxelParamsGPU), &params);

        // One invocation per CELL (voxelDims, not sampleDims) - every
        // cell is dispatched, but voxel_march.comp's first real line is
        // a dirty-chunk check that early-exits any cell whose chunk
        // isn't in the set we just flagged above. Simple-and-correct
        // over cleverness: computing a tight per-dirty-chunk dispatch
        // region on the CPU would need per-chunk vkCmdDispatchBase
        // calls (one per dirty chunk) instead of one global dispatch -
        // real added complexity to save work on cells that instantly
        // early-exit anyway; not worth it until march is happening
        // often enough (rapid carving) for that overhead to show up.
        glm::ivec3 voxelDims = volume.desc.VoxelDims();
        uint32_t cellCount = static_cast<uint32_t>(voxelDims.x * voxelDims.y * voxelDims.z);
        vkCmdDispatch(cmd, DispatchGroupCount(cellCount), 1, 1);

        // Clear dirty flags back to 0 for the same chunks we set above -
        // this march fully processed them, they shouldn't stay flagged.
        VkMemoryBarrier toTransfer2{};
        toTransfer2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toTransfer2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toTransfer2.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &toTransfer2, 0, nullptr, 0, nullptr);
        for (uint32_t chunkIndex : volume.dirtyChunkIndices)
        {
            vkCmdFillBuffer(cmd, volume.dirtyFlags.Get(), static_cast<VkDeviceSize>(chunkIndex) * sizeof(uint32_t), sizeof(uint32_t), 0);
        }

        // Compute-write (vertices/indirect) -> indirect-draw-read and
        // vertex-shader-read, both of which happen in Renderer::
        // DrawFrame well after this call returns (this function blocks
        // on a fence tied to this command buffer below before returning -
        // see SubmitAndWaitForCommandBuffer's own comment - so this
        // barrier is strictly about intra-command-buffer ordering here,
        // not cross-queue sync - matches ParticleSystemGPU's pattern of
        // barriers-within-the-recorded-work).
        VkMemoryBarrier toIndirectAndVertex{};
        toIndirectAndVertex.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toIndirectAndVertex.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        toIndirectAndVertex.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                              0, 1, &toIndirectAndVertex, 0, nullptr, 0, nullptr);

        vkEndCommandBuffer(cmd);
        SubmitAndWaitForCommandBuffer(cmd);

        // Summary count only - this used to also print every individual
        // chunk index in the batch, which was harmless for a handful of
        // chunks but became real, unnecessary overhead once a single
        // march started covering thousands of chunks (large reform/melt
        // volumes near kMaxSamplesPerAxis - see ReformSystem.cpp's own
        // comment on that cap): thousands of individual std::printf
        // calls per march, purely diagnostic, with no value at that
        // volume - nobody reads 2500 individual chunk numbers. Removed
        // rather than gated behind a verbose flag since the per-chunk
        // list was never actually useful even at small counts, just
        // harmless there.
        std::printf("[March]  re-marched %zu chunk(s)\n", volume.dirtyChunkIndices.size());

        volume.dirtyChunkIndices.clear();
    }

    void VoxelSystemGPU::DiscardPendingMarch(VoxelVolumeHandle handle)
    {
        // No GPU work at all - see this function's own header comment
        // (VoxelSystemGPU.h) for why that's correct, not just fast, for
        // a raymarch-only volume. The density buffer is already correct
        // on the GPU (Carve()'s UploadDensityRegion uploaded it directly,
        // same as SeedBox's UploadDensity does at registration) -
        // dirtyChunkIndices exists purely to tell MarchDirtyChunks which
        // chunks' MESH needs regenerating, and nothing here has a mesh
        // consumer to regenerate it for.
        FlushPendingUpload(handle); // raymarch-only volumes render straight from the GPU density buffer - it must be current
        Volume& volume = GetVolume(handle);
        volume.dirtyChunkIndices.clear();
    }

    void VoxelSystemGPU::Carve(VoxelVolumeHandle handle, const glm::vec3& worldPos, float radius)
    {
        EDEN_PROFILE_SCOPE("Voxel: Carve() CPU density edit");
        Volume& volume = GetVolume(handle);
        glm::vec3 localCenter = worldPos - volume.desc.origin;

        // Which chunks could contain a cell affected by this carve -
        // padded by one voxel on every side of the carve's own AABB so
        // a boundary sample shared between two chunks (see
        // VoxelSystemGPU.h's class comment on shared boundary samples)
        // always marks BOTH sides dirty, not just whichever chunk the
        // math happens to round into. A chunk or two of unnecessary
        // re-marching at the edge of the affected region is cheap;
        // missing a chunk that should have updated is a visible seam.
        glm::vec3 voxelSize3(volume.desc.voxelSize);
        glm::vec3 localMin = (localCenter - glm::vec3(radius)) / voxelSize3 - glm::vec3(1.0f);
        glm::vec3 localMax = (localCenter + glm::vec3(radius)) / voxelSize3 + glm::vec3(1.0f);

        glm::ivec3 voxelDims = volume.desc.VoxelDims();
        glm::ivec3 cellMin = glm::clamp(glm::ivec3(glm::floor(localMin)), glm::ivec3(0), voxelDims - glm::ivec3(1));
        glm::ivec3 cellMax = glm::clamp(glm::ivec3(glm::ceil(localMax)), glm::ivec3(0), voxelDims - glm::ivec3(1));

        // BUGFIX: chunkMax must be derived from the actual touched SAMPLE
        // range (cellMax + 1, since a carved cell's far corner sample
        // belongs to the NEXT cell over), not from cellMax directly.
        // Whenever cellMax lands on a chunk's last cell (e.g. cell 7 of
        // an 8-cell chunk), the carve still writes into sample 8 - the
        // boundary sample shared with the next chunk's first cell - but
        // the old `ChunkCoordFromCellClamped(cellMax)` call never saw
        // past cell 7, so that neighboring chunk was never marked dirty
        // despite one of its cells' corner densities having just
        // changed. Its mesh kept stale (pre-carve) geometry at exactly
        // that shared face while the touched chunk's mesh updated,
        // producing a seam/hole with no closing wall on one side. Using
        // cellMax + 1 (clamped) here matches the +1 padding sampleMax
        // already uses below, so the chunk range and sample range agree.
        //
        // BUGFIX 2: chunkMin needs the EXACT mirror-image fix, previously
        // missing entirely - sampleMin is the first touched sample, and
        // that same sample is the FAR corner of the cell just BELOW
        // cellMin (cell sampleMin-1 spans samples sampleMin-1 and
        // sampleMin). Deriving chunkMin from cellMin directly missed
        // that cell's chunk whenever sampleMin landed exactly on a chunk
        // boundary - the asymmetric version of the exact bug described
        // above, just on the low side instead of the high side. This was
        // real and reproducible, not theoretical: a carve with
        // cellMin=(16,...) logged chunkMin=(2,...) (chunk 10 only) even
        // though sample x=16 is chunk 9's last cell's far corner - chunk
        // 9 never got marked dirty, leaving a permanent stale seam
        // exactly at that boundary. cellMin - 1 (clamped) mirrors
        // cellMax + 1 above.
        glm::ivec3 chunkMin = ChunkCoordFromCellClamped(volume.desc, glm::max(cellMin - glm::ivec3(1), glm::ivec3(0)));
        glm::ivec3 chunkMax = ChunkCoordFromCellClamped(volume.desc, glm::min(cellMax + glm::ivec3(1), voxelDims - glm::ivec3(1)));

        for (int32_t cz = chunkMin.z; cz <= chunkMax.z; ++cz)
        for (int32_t cy = chunkMin.y; cy <= chunkMax.y; ++cy)
        for (int32_t cx = chunkMin.x; cx <= chunkMax.x; ++cx)
        {
            int32_t chunkIndex = cx + cy * volume.desc.chunkDims.x + cz * volume.desc.chunkDims.x * volume.desc.chunkDims.y;
            volume.chunkSolid[chunkIndex] = false;
            if (std::find(volume.dirtyChunkIndices.begin(), volume.dirtyChunkIndices.end(), static_cast<uint32_t>(chunkIndex))
                == volume.dirtyChunkIndices.end())
            {
                volume.dirtyChunkIndices.push_back(static_cast<uint32_t>(chunkIndex));
            }
        }

        // The carve computation itself only touches the affected SAMPLE
        // region (padded by 1 voxel, same margin reasoning as the chunk
        // range above), not the whole field - unlike the old GPU-
        // dispatch version this replaced, which ran over every sample
        // and relied on the shader to early-exit ones outside the carve
        // radius. A bounded CPU loop over just the relevant samples is
        // both simpler and cheaper here.
        glm::ivec3 sampleDims = volume.desc.SampleDims();
        glm::ivec3 sampleMin = glm::clamp(cellMin, glm::ivec3(0), sampleDims - glm::ivec3(1));
        glm::ivec3 sampleMax = glm::clamp(cellMax + glm::ivec3(1), glm::ivec3(0), sampleDims - glm::ivec3(1));

        int32_t flippedSolidToEmpty = 0;
        int32_t stillSolid = 0;

        for (int32_t z = sampleMin.z; z <= sampleMax.z; ++z)
        for (int32_t y = sampleMin.y; y <= sampleMax.y; ++y)
        for (int32_t x = sampleMin.x; x <= sampleMax.x; ++x)
        {
            glm::vec3 localPos = glm::vec3(x, y, z) * volume.desc.voxelSize;
            float carveSDF = glm::length(localPos - localCenter) - radius;
            uint32_t idx = SampleIndex(volume.desc, glm::ivec3(x, y, z));
            // CSG subtraction: A - B = max(A, -B). Where the carve
            // sphere overlaps solid (density negative, carveSDF negative
            // inside the tool), -carveSDF is positive there, pushing
            // max() toward "empty". Outside the carve sphere, -carveSDF
            // is very negative and max() leaves the original density
            // untouched.
            bool wasSolid = volume.densityCPU[idx] < kVoxelIsoLevel;
            volume.densityCPU[idx] = std::max(volume.densityCPU[idx], -carveSDF);
            bool isSolidNow = volume.densityCPU[idx] < kVoxelIsoLevel;
            if (wasSolid && !isSolidNow) flippedSolidToEmpty++;
            else if (wasSolid && isSolidNow) stillSolid++;
        }

        // Verbose per-carve diagnostics - OFF by default. This used to
        // run unconditionally, 3-4 std::printf calls on EVERY Carve()
        // invocation. That was tolerable back when carving was one click
        // = one carve; it stopped being tolerable once carving could be
        // held (10 ticks/sec) and a single tick can mean multiple Carve()
        // calls (terrain-tile-boundary wraparound alone doubles it - see
        // the paired entries in a carve log near a tile seam). Synchronous
        // console writes are genuinely slow on plenty of terminals/IDE
        // consoles - 80+ of them a second while holding the carve button
        // down is real, measurable stutter, not free logging. Flip this
        // to true for one session if this needs re-diagnosing; nothing
        // else needs to change to get the diagnostics back.
        constexpr bool kCarveDebugLogging = false;
        if constexpr (kCarveDebugLogging)
        {
            std::printf("[Carve] world=(%.3f,%.3f,%.3f) local=(%.3f,%.3f,%.3f) radius=%.3f\n",
                        worldPos.x, worldPos.y, worldPos.z, localCenter.x, localCenter.y, localCenter.z, radius);
            std::printf("[Carve]   cellMin=(%d,%d,%d) cellMax=(%d,%d,%d) sampleMin=(%d,%d,%d) sampleMax=(%d,%d,%d)\n",
                        cellMin.x, cellMin.y, cellMin.z, cellMax.x, cellMax.y, cellMax.z,
                        sampleMin.x, sampleMin.y, sampleMin.z, sampleMax.x, sampleMax.y, sampleMax.z);
            std::printf("[Carve]   chunkMin=(%d,%d,%d) chunkMax=(%d,%d,%d) dirtyChunkCount=%zu flippedSolidToEmpty=%d stillSolid=%d\n",
                        chunkMin.x, chunkMin.y, chunkMin.z, chunkMax.x, chunkMax.y, chunkMax.z,
                        volume.dirtyChunkIndices.size(), flippedSolidToEmpty, stillSolid);
        }

        // Extra diagnostic: was this carve's own sample range clamped
        // against the volume boundary at all (i.e. did the request reach
        // past the edge of the density array)? If so, print by how much -
        // this tells us whether "boundary proximity" is actually in play
        // for this specific carve, rather than inferring it indirectly
        // from cellMax looking suspiciously close to voxelDims-1.
        glm::ivec3 requestedMin = glm::ivec3(glm::floor(localMin));
        glm::ivec3 requestedMax = glm::ivec3(glm::ceil(localMax)) + glm::ivec3(1);
        if constexpr (kCarveDebugLogging)
        {
            bool clampedLow = glm::any(glm::lessThan(requestedMin, glm::ivec3(0)));
            bool clampedHigh = glm::any(glm::greaterThanEqual(requestedMax, sampleDims));
            if (clampedLow || clampedHigh)
            {
                std::printf("[Carve]   ** BOUNDARY CLAMPED ** requestedMin=(%d,%d,%d) requestedMax=(%d,%d,%d) sampleDims=(%d,%d,%d)\n",
                            requestedMin.x, requestedMin.y, requestedMin.z,
                            requestedMax.x, requestedMax.y, requestedMax.z,
                            sampleDims.x, sampleDims.y, sampleDims.z);
            }
        }

        // Union into the pending region RecomputeExactDistances will
        // consume - see Volume::hasPendingExactDistanceRegion's own
        // comment. Unioning (not overwriting) is what makes this correct
        // across a drill-through stroke's multiple Carve() calls on the
        // same volume before RecomputeExactDistances runs once at the end.
        if (volume.hasPendingExactDistanceRegion)
        {
            volume.pendingExactDistanceMin = glm::min(volume.pendingExactDistanceMin, sampleMin);
            volume.pendingExactDistanceMax = glm::max(volume.pendingExactDistanceMax, sampleMax);
        }
        else
        {
            volume.pendingExactDistanceMin = sampleMin;
            volume.pendingExactDistanceMax = sampleMax;
            volume.hasPendingExactDistanceRegion = true;
        }

        // Deferred - see Volume::hasPendingUpload. Union with anything
        // already pending (several Carve() calls per gesture).
        if (volume.hasPendingUpload)
        {
            volume.pendingUploadMin = glm::min(volume.pendingUploadMin, sampleMin);
            volume.pendingUploadMax = glm::max(volume.pendingUploadMax, sampleMax);
        }
        else
        {
            volume.pendingUploadMin = sampleMin;
            volume.pendingUploadMax = sampleMax;
            volume.hasPendingUpload = true;
        }
    }

    void VoxelSystemGPU::FlushPendingUpload(VoxelVolumeHandle handle)
    {
        if (!IsValid(handle))
        {
            return;
        }
        Volume& volume = GetVolume(handle);
        if (!volume.hasPendingUpload || volume.densityCPU.empty())
        {
            return;
        }
        volume.hasPendingUpload = false; // cleared first: UploadDensityRegion can throw, and a retry loop on a bad region helps nobody
        UploadDensityRegion(volume, volume.pendingUploadMin, volume.pendingUploadMax);
    }

    void VoxelSystemGPU::FlushAllPendingUploads()
    {
        for (Volume& volume : m_Volumes)
        {
            // Freed slots have densityCPU swapped empty (see UnregisterVolume)
            // and hasPendingUpload reset; the checks are belt and braces.
            if (volume.hasPendingUpload && !volume.densityCPU.empty())
            {
                volume.hasPendingUpload = false;
                UploadDensityRegion(volume, volume.pendingUploadMin, volume.pendingUploadMax);
            }
        }
    }

    // REFERENCE implementation of RecomputeExactDistances - the original
    // whole-volume algorithm, kept ONLY so the optimized version can be
    // checked against it (set EDEN_VERIFY_DISTANCE=1; see
    // RecomputeExactDistances). Not used in normal runs.
    static void ReferenceExactDistances(const VoxelVolumeDesc& desc, std::vector<float>& density, glm::ivec3 boxMin, glm::ivec3 boxMax)
    {
        glm::ivec3 sampleDims = desc.SampleDims();
        // Same math as the private VoxelSystemGPU::SampleIndex (this helper is a
        // free function so it can't call it).
        auto SampleIndexRef = [&desc](const VoxelVolumeDesc&, glm::ivec3 sample) -> uint32_t
        {
            glm::ivec3 dims = desc.SampleDims();
            glm::ivec3 clamped = glm::clamp(sample, glm::ivec3(0), dims - glm::ivec3(1));
            return static_cast<uint32_t>(clamped.x) + static_cast<uint32_t>(clamped.y) * static_cast<uint32_t>(dims.x)
                 + static_cast<uint32_t>(clamped.z) * static_cast<uint32_t>(dims.x) * static_cast<uint32_t>(dims.y);
        };
        size_t sampleCount = static_cast<size_t>(sampleDims.x) * sampleDims.y * sampleDims.z;
        struct Neighbor { glm::ivec3 offset; float length; };
        static const std::vector<Neighbor> kNeighbors = []
        {
            std::vector<Neighbor> result;
            for (int32_t dz = -1; dz <= 1; ++dz)
            for (int32_t dy = -1; dy <= 1; ++dy)
            for (int32_t dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                glm::ivec3 offset(dx, dy, dz);
                result.push_back({ offset, glm::length(glm::vec3(offset)) });
            }
            return result;
        }();
        std::vector<float> trueDistance(sampleCount, std::numeric_limits<float>::infinity());
        std::vector<bool> finalized(sampleCount, false);

        using QueueEntry = std::pair<float, uint32_t>; // (distance, sample index)
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;

        // Seeds from EVERY solid sample in the WHOLE field, not just the
        // padded box - a solid sample just outside the box can still be
        // the nearest neighbor for a non-solid sample right at the box's
        // edge, and this scan is cheap (a single bandwidth-bound pass,
        // not the priority-queue traversal that was the actual cost -
        // see this function's header comment).
        for (size_t i = 0; i < sampleCount; ++i)
        {
            if (density[i] < kVoxelIsoLevel)
            {
                trueDistance[i] = 0.0f;
                queue.push({ 0.0f, static_cast<uint32_t>(i) });
            }
        }

        // Index <-> coordinate, matching SampleIndex's own packing
        // exactly (x + y*dims.x + z*dims.x*dims.y) - see that function's
        // comment for why this layout is the one place it's defined.
        auto CoordFromIndex = [sampleDims](uint32_t index) -> glm::ivec3
        {
            int32_t x = static_cast<int32_t>(index) % sampleDims.x;
            int32_t y = (static_cast<int32_t>(index) / sampleDims.x) % sampleDims.y;
            int32_t z = static_cast<int32_t>(index) / (sampleDims.x * sampleDims.y);
            return glm::ivec3(x, y, z);
        };

        while (!queue.empty())
        {
            auto [dist, index] = queue.top();
            queue.pop();
            if (finalized[index])
            {
                continue; // a cheaper path to this sample was already finalized
            }
            finalized[index] = true;

            glm::ivec3 coord = CoordFromIndex(index);
            for (const Neighbor& neighbor : kNeighbors)
            {
                glm::ivec3 neighborCoord = coord + neighbor.offset;
                if (glm::any(glm::lessThan(neighborCoord, glm::ivec3(0))) ||
                    glm::any(glm::greaterThanEqual(neighborCoord, sampleDims)))
                {
                    continue;
                }

                // The actual bound: never propagate INTO a cell outside
                // the padded box. A solid seed sitting outside the box
                // still gets popped and still checks its neighbors (that's
                // how it correctly seeds a box-edge cell one hop inside),
                // but nothing outside the box ever gets pushed onward from
                // there - so the traversal explores at most (every solid
                // sample's one-hop neighborhood) + (the padded box itself),
                // never the whole non-solid field. That first term is why
                // the seed scan above still has to be global rather than
                // box-limited too.
                if (glm::any(glm::lessThan(neighborCoord, boxMin)) ||
                    glm::any(glm::greaterThan(neighborCoord, boxMax)))
                {
                    continue;
                }

                uint32_t neighborIndex = SampleIndexRef(desc, neighborCoord);
                if (finalized[neighborIndex])
                {
                    continue;
                }

                float candidateDist = dist + neighbor.length * desc.voxelSize;
                if (candidateDist < trueDistance[neighborIndex])
                {
                    trueDistance[neighborIndex] = candidateDist;
                    queue.push({ candidateDist, neighborIndex });
                }
            }
        }

        // Only overwrite non-solid samples INSIDE the padded box - solid
        // samples already hold exact, locally-correct CSG values from
        // SeedSphere/Carve and were never the source of this bug (same as
        // before). Samples outside the box were never relaxed this call
        // (trueDistance[i] is still its infinity initial value for most
        // of them), so writing them back would OVERWRITE an already-
        // correct value from a previous call with garbage - skipping them
        // entirely is what makes bounding this to a region safe, not just
        // fast.
        for (int32_t z = boxMin.z; z <= boxMax.z; ++z)
        for (int32_t y = boxMin.y; y <= boxMax.y; ++y)
        for (int32_t x = boxMin.x; x <= boxMax.x; ++x)
        {
            uint32_t i = SampleIndexRef(desc, glm::ivec3(x, y, z));
            if (density[i] >= kVoxelIsoLevel)
            {
                density[i] = trueDistance[i];
            }
        }
    }

    void VoxelSystemGPU::RecomputeExactDistances(VoxelVolumeHandle handle)
    {
        EDEN_PROFILE_SCOPE("Voxel: exact distance transform (CPU)");
        // Multi-source Dijkstra distance transform, seeded from every
        // currently-solid sample - see this function's header comment
        // for WHY this exists (Carve() alone leaves every untouched
        // sample holding a stale, systematically-too-small distance).
        // This computes the TRUE distance from every non-solid sample to
        // the nearest CURRENTLY-solid sample, with no dependency on
        // carve radius or coverage - correctness here comes from
        // actually searching the field, not from hoping every carve
        // stroke overlapped enough.
        //
        // BOUNDED to a padded region around what Carve() actually
        // touched (see Volume::hasPendingExactDistanceRegion's own
        // comment) - this was briefly reverted to full-field on a theory
        // that partial coverage caused a chunk-boundary mesh seam a user
        // reported. That theory was WRONG (confirmed: this function only
        // ever writes to densityCPU and never re-uploads to the GPU or
        // triggers a re-march, so nothing it does can reach the rendered
        // mesh at all - see the bottom of this function), and reverting
        // it cost real, measured performance (lag roughly doubled) for
        // zero correctness benefit, so it's back. The actual seam cause
        // is still open and unrelated to this function.
        //
        // Bounding this at all exists because the un-bounded version was
        // the actual cause of a real multi-second freeze carving anything
        // fine-grained (a mold workpiece's much finer voxelSize means
        // vastly more samples than a terrain tile - the exact same
        // O(wholeField log wholeField) algorithm scales with sample
        // count, not carve size). A local carve can only have changed
        // which sample is nearest-solid for points near where it
        // happened - kPaddingVoxels below is deliberately generous
        // relative to typical carve radii so this stays correct in
        // practice, not just fast; see its own comment for the actual
        // trade-off being made.
        static const bool verifyAgainstReference = (std::getenv("EDEN_VERIFY_DISTANCE") != nullptr);
        Volume& volume = GetVolume(handle);
        glm::ivec3 sampleDims = volume.desc.SampleDims();
        size_t sampleCount = static_cast<size_t>(sampleDims.x) * sampleDims.y * sampleDims.z;

        // Consumed here - the NEXT Carve() call starts a fresh pending
        // region. Falls back to the whole field (this function's
        // original, always-correct behavior) if nothing is pending, which
        // only happens if something other than the carve-input path ever
        // calls this directly - see this function's only current caller
        // in ProcessVoxelCarveInput.
        glm::ivec3 boxMin(0);
        glm::ivec3 boxMax = sampleDims - glm::ivec3(1);
        if (volume.hasPendingExactDistanceRegion)
        {
            constexpr int32_t kPaddingVoxels = 12; // see this function's header comment - comfortably larger than sampleMin/sampleMax's own already-tight carve-radius span, not a tuned-to-the-wire minimum
            boxMin = glm::max(volume.pendingExactDistanceMin - glm::ivec3(kPaddingVoxels), glm::ivec3(0));
            boxMax = glm::min(volume.pendingExactDistanceMax + glm::ivec3(kPaddingVoxels), sampleDims - glm::ivec3(1));
        }
        volume.hasPendingExactDistanceRegion = false;

        std::vector<float> referenceDensity;
        if (verifyAgainstReference)
        {
            referenceDensity = volume.densityCPU;
        }

        // 26-connected neighbor offsets (face/edge/corner), each with
        // its true Euclidean length so the propagated distance is a real
        // (grid-quantized, but not axis-biased) distance, not a cheaper
        // Manhattan/Chebyshev approximation that would visibly distort
        // near diagonals.
        struct Neighbor { glm::ivec3 offset; float length; };
        static const std::vector<Neighbor> kNeighbors = []
        {
            std::vector<Neighbor> result;
            for (int32_t dz = -1; dz <= 1; ++dz)
            for (int32_t dy = -1; dy <= 1; ++dy)
            for (int32_t dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                glm::ivec3 offset(dx, dy, dz);
                result.push_back({ offset, glm::length(glm::vec3(offset)) });
            }
            return result;
        }();

        // OPTIMIZED, result-identical to the original (ReferenceExactDistances):
        //  * The original allocated and scanned arrays for the WHOLE volume and
        //    pushed EVERY solid sample into the priority queue, then spent most
        //    of its time popping seeds whose neighbors it immediately rejected
        //    for being outside the box. Only the box matters: nothing outside
        //    it is ever relaxed, so the only outside samples that can influence
        //    the result are solid seeds directly adjacent to the box - hence
        //    the one-sample border on `region`.
        //  * A solid seed whose in-box neighbors are all solid can't improve
        //    anything (every candidate would be > 0 against an existing 0), so
        //    only seeds touching at least one non-solid in-box sample are queued.
        const glm::ivec3 regionMin = glm::max(boxMin - glm::ivec3(1), glm::ivec3(0));
        const glm::ivec3 regionMax = glm::min(boxMax + glm::ivec3(1), sampleDims - glm::ivec3(1));
        const glm::ivec3 regionDims = regionMax - regionMin + glm::ivec3(1);
        const size_t regionCount = static_cast<size_t>(regionDims.x) * regionDims.y * regionDims.z;

        auto LocalIndex = [&](const glm::ivec3& c) -> uint32_t
        {
            glm::ivec3 l = c - regionMin;
            return static_cast<uint32_t>(l.x + regionDims.x * (l.y + regionDims.y * l.z));
        };
        auto InBox = [&](const glm::ivec3& c)
        {
            return c.x >= boxMin.x && c.y >= boxMin.y && c.z >= boxMin.z &&
                   c.x <= boxMax.x && c.y <= boxMax.y && c.z <= boxMax.z;
        };

        std::vector<float> trueDistance(regionCount, std::numeric_limits<float>::infinity());
        std::vector<uint8_t> finalized(regionCount, 0);

        using QueueEntry = std::pair<float, uint32_t>; // (distance, LOCAL region index)
        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;

        // Pass 1: mark solids (distance 0) and queue only the ones that touch a non-solid in-box sample.
        for (int32_t z = regionMin.z; z <= regionMax.z; ++z)
        for (int32_t y = regionMin.y; y <= regionMax.y; ++y)
        for (int32_t x = regionMin.x; x <= regionMax.x; ++x)
        {
            const glm::ivec3 coord(x, y, z);
            if (!(volume.densityCPU[SampleIndex(volume.desc, coord)] < kVoxelIsoLevel))
            {
                continue;
            }
            const uint32_t local = LocalIndex(coord);
            trueDistance[local] = 0.0f;

            bool touchesOpen = false;
            for (const Neighbor& neighbor : kNeighbors)
            {
                const glm::ivec3 nc = coord + neighbor.offset;
                if (!InBox(nc) || nc.x >= sampleDims.x || nc.y >= sampleDims.y || nc.z >= sampleDims.z)
                {
                    continue;
                }
                if (!(volume.densityCPU[SampleIndex(volume.desc, nc)] < kVoxelIsoLevel))
                {
                    touchesOpen = true;
                    break;
                }
            }
            if (touchesOpen)
            {
                queue.push({ 0.0f, local });
            }
        }

        while (!queue.empty())
        {
            const auto [dist, local] = queue.top();
            queue.pop();
            if (finalized[local])
            {
                continue;
            }
            finalized[local] = 1;

            const glm::ivec3 coord(regionMin.x + static_cast<int32_t>(local % regionDims.x),
                                    regionMin.y + static_cast<int32_t>((local / regionDims.x) % regionDims.y),
                                    regionMin.z + static_cast<int32_t>(local / (regionDims.x * regionDims.y)));
            for (const Neighbor& neighbor : kNeighbors)
            {
                const glm::ivec3 neighborCoord = coord + neighbor.offset;
                if (!InBox(neighborCoord) || glm::any(glm::greaterThanEqual(neighborCoord, sampleDims)))
                {
                    continue; // outside the padded box (InBox is already >= 0 clamped via boxMin)
                }
                const uint32_t neighborLocal = LocalIndex(neighborCoord);
                if (finalized[neighborLocal])
                {
                    continue;
                }
                const float candidateDist = dist + neighbor.length * volume.desc.voxelSize;
                if (candidateDist < trueDistance[neighborLocal])
                {
                    trueDistance[neighborLocal] = candidateDist;
                    queue.push({ candidateDist, neighborLocal });
                }
            }
        }

        for (int32_t z = boxMin.z; z <= boxMax.z; ++z)
        for (int32_t y = boxMin.y; y <= boxMax.y; ++y)
        for (int32_t x = boxMin.x; x <= boxMax.x; ++x)
        {
            const uint32_t i = SampleIndex(volume.desc, glm::ivec3(x, y, z));
            if (volume.densityCPU[i] >= kVoxelIsoLevel)
            {
                volume.densityCPU[i] = trueDistance[LocalIndex(glm::ivec3(x, y, z))];
            }
        }

        // Optional self-check against the original algorithm. Run with
        // EDEN_VERIFY_DISTANCE=1 and watch the console: it prints the largest
        // difference over the box (0 expected - same graph, same weights).
        // `referenceDensity` was copied BEFORE the optimized pass touched
        // anything.
        if (!referenceDensity.empty())
        {
            ReferenceExactDistances(volume.desc, referenceDensity, boxMin, boxMax);
            float maxDiff = 0.0f;
            size_t mismatches = 0;
            for (int32_t z = boxMin.z; z <= boxMax.z; ++z)
            for (int32_t y = boxMin.y; y <= boxMax.y; ++y)
            for (int32_t x = boxMin.x; x <= boxMax.x; ++x)
            {
                const uint32_t i = SampleIndex(volume.desc, glm::ivec3(x, y, z));
                const float a = volume.densityCPU[i];
                const float b = referenceDensity[i];
                const float diff = std::isinf(a) && std::isinf(b) ? 0.0f : std::fabs(a - b);
                if (diff > 1e-4f) { ++mismatches; }
                maxDiff = std::max(maxDiff, diff);
            }
            std::printf("[VerifyDistance] box=(%d,%d,%d)-(%d,%d,%d) maxDiff=%.6f mismatches=%zu\n",
                        boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z, maxDiff, mismatches);
        }

        // Deliberately no GPU re-upload and no MarchDirtyChunks call
        // here - every value this touches stays on the same (non-solid)
        // side of the iso level it was already on, so the marched mesh
        // is bit-for-bit unaffected. This only matters to
        // SampleSignedDistance/SampleGradient (collision), which read
        // densityCPU directly - see that function's own comment.
    }

    void VoxelSystemGPU::DebugScanAmbiguousCells(VoxelVolumeHandle handle) const
    {
        const Volume& volume = GetVolume(handle);
        glm::ivec3 voxelDims = volume.desc.VoxelDims();

        // Same corner numbering/offsets voxel_march.comp uses
        // (kCornerOffset) - kept identical so "corner i" means the same
        // physical corner on both the CPU scan and the GPU march.
        static const glm::ivec3 kCornerOffset[8] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };

        // Each cube face as 4 corner indices, in ring order around the
        // face (so index i and i+2 are the face's two diagonals). Faces:
        // -Z, +Z, -Y, +Y, -X, +X.
        static const int kFaces[6][4] = {
            {0,1,2,3}, {4,5,6,7}, {0,1,5,4}, {3,2,6,7}, {0,3,7,4}, {1,2,6,5}
        };

        int32_t ambiguousCount = 0;
        int32_t scannedCount = 0;

        for (int32_t z = 0; z < voxelDims.z; ++z)
        for (int32_t y = 0; y < voxelDims.y; ++y)
        for (int32_t x = 0; x < voxelDims.x; ++x)
        {
            glm::ivec3 cell(x, y, z);
            float d[8];
            bool solid[8];
            for (int i = 0; i < 8; ++i)
            {
                uint32_t idx = SampleIndex(volume.desc, cell + kCornerOffset[i]);
                d[i] = volume.densityCPU[idx];
                solid[i] = d[i] < kVoxelIsoLevel;
            }

            // Skip fully-uniform cells (no surface passes through here
            // at all, ambiguity is moot) - same edgeTable==0 early-out
            // the shader itself uses, just computed directly from solid[].
            bool anySolid = solid[0]||solid[1]||solid[2]||solid[3]||solid[4]||solid[5]||solid[6]||solid[7];
            bool allSolid = solid[0]&&solid[1]&&solid[2]&&solid[3]&&solid[4]&&solid[5]&&solid[6]&&solid[7];
            if (!anySolid || allSolid)
            {
                continue;
            }
            scannedCount++;

            bool cellIsAmbiguous = false;
            int ambiguousFace = -1;
            for (int f = 0; f < 6 && !cellIsAmbiguous; ++f)
            {
                bool a = solid[kFaces[f][0]];
                bool b = solid[kFaces[f][1]];
                bool c = solid[kFaces[f][2]];
                bool e = solid[kFaces[f][3]];
                // Face ambiguity: one diagonal (a,c) agrees with each
                // other but DISagrees with the other diagonal (b,e),
                // which also agrees with each other - i.e. a==c, b==e,
                // a!=b. That's the "solid corners only on one diagonal"
                // pattern with no direct edge connecting them.
                if (a == c && b == e && a != b)
                {
                    cellIsAmbiguous = true;
                    ambiguousFace = f;
                }
            }

            // Real correctness check, not just a symptom count: compute
            // the SAME cellIndex bit pattern voxel_march.comp does
            // (Lewiner's own convention - bit set when density is ABOVE
            // isoLevel, the OPPOSITE of this function's own `solid[]`,
            // which is why this isn't just `!solid[i]`) and look it up
            // in the actual MC33 CASES table this build uploads to the
            // GPU. Every cell reaching this point already has at least
            // one solid and one non-solid corner (the anySolid/allSolid
            // filter above), which makes case<=0 here IMPOSSIBLE if the
            // lookup is wired correctly - CASES only returns <=0 for the
            // two fully-uniform configs (0 and 255), both already
            // excluded. If this ever fires, it's conclusive proof of an
            // indexing/convention bug in the MC33 port (wrong bit
            // convention, wrong table offset, etc.) - not a case of
            // "normal ambiguity," which this check can't produce a false
            // positive for.
            int cellIndex = 0;
            for (int i = 0; i < 8; ++i)
            {
                if (d[i] > kVoxelIsoLevel)
                {
                    cellIndex |= (1 << i);
                }
            }
            int mc33Case = kMC33TableData[kMC33Offset_CASES + cellIndex * 2 + 0];
            if (mc33Case <= 0)
            {
                std::printf("[MC33_BUG] cell=(%d,%d,%d) cellIndex=%d got case=%d despite being a genuine "
                            "surface crossing - this is a real indexing bug, not expected ambiguity\n",
                            x, y, z, cellIndex, mc33Case);
            }

            if (cellIsAmbiguous)
            {
                ambiguousCount++;
                std::printf("[AmbiguousCell] cell=(%d,%d,%d) face=%d cellIndex=%d mc33Case=%d corners_solid=[%d,%d,%d,%d,%d,%d,%d,%d] density=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                            x, y, z, ambiguousFace, cellIndex, mc33Case,
                            solid[0],solid[1],solid[2],solid[3],solid[4],solid[5],solid[6],solid[7],
                            d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
            }
        }

        std::printf("[AmbiguousScan] surface-crossing cells=%d ambiguous=%d (%.1f%%)\n",
                    scannedCount, ambiguousCount, scannedCount > 0 ? 100.0f * ambiguousCount / scannedCount : 0.0f);
    }

    bool VoxelSystemGPU::IsWorldPointSolid(VoxelVolumeHandle handle, const glm::vec3& worldPos) const
    {
        const Volume& volume = GetVolume(handle);
        glm::vec3 localPos = worldPos - volume.desc.origin;
        glm::ivec3 voxelDims = volume.desc.VoxelDims();
        glm::vec3 cellF = localPos / volume.desc.voxelSize;

        if (glm::any(glm::lessThan(cellF, glm::vec3(0.0f))) || glm::any(glm::greaterThanEqual(cellF, glm::vec3(voxelDims))))
        {
            return false; // outside the volume entirely - see header comment on this default
        }

        glm::ivec3 cell = glm::ivec3(cellF);
        int32_t chunkIndex = ChunkIndexForCell(volume.desc, cell);
        return volume.chunkSolid[chunkIndex];
    }

    bool VoxelSystemGPU::IsWorldPointSolidExact(VoxelVolumeHandle handle, const glm::vec3& worldPos) const
    {
        const Volume& volume = GetVolume(handle);
        glm::vec3 localPos = worldPos - volume.desc.origin;
        glm::ivec3 voxelDims = volume.desc.VoxelDims();
        glm::vec3 cellF = localPos / volume.desc.voxelSize;

        // Same out-of-bounds convention as IsWorldPointSolid - a point
        // outside the volume's sample grid entirely never reads solid.
        if (glm::any(glm::lessThan(cellF, glm::vec3(0.0f))) || glm::any(glm::greaterThanEqual(cellF, glm::vec3(voxelDims))))
        {
            return false;
        }

        return SampleDensityTrilinear(volume, localPos) < kVoxelIsoLevel;
    }

    float VoxelSystemGPU::SampleSignedDistance(VoxelVolumeHandle handle, const glm::vec3& worldPos) const
    {
        const Volume& volume = GetVolume(handle);
        glm::vec3 localPos = worldPos - volume.desc.origin;

        // SampleDensityTrilinear clamps its query point to the sample
        // grid internally (see its own comment) - a point genuinely
        // outside the volume's bounds would otherwise just read back the
        // boundary sample's value flatly, understating how far outside
        // it actually is. Clamp here first and add back the exact
        // Euclidean distance from that clamp, same "box distance"
        // identity SDF::Box uses - for a point already inside the grid
        // this added term is zero, so this changes nothing for the case
        // narrow phase actually cares about (a sphere near/at the
        // volume's surface, which is all broad phase would have let
        // through anyway).
        glm::vec3 extent = glm::vec3(volume.desc.VoxelDims()) * volume.desc.voxelSize;
        glm::vec3 clampedLocal = glm::clamp(localPos, glm::vec3(0.0f), extent);
        float outsideDistance = glm::length(localPos - clampedLocal);

        return SampleDensityTrilinear(volume, clampedLocal) + outsideDistance;
    }

    glm::vec3 VoxelSystemGPU::SampleGradient(VoxelVolumeHandle handle, const glm::vec3& worldPos) const
    {
        // Tetrahedron-technique gradient (Inigo Quilez's calcNormal
        // trick, standard in SDF raymarching) - 4 SampleSignedDistance
        // taps instead of the naive central-difference's 6. Each tap is
        // offset toward one vertex of a regular tetrahedron centered on
        // worldPos, weighted-summed by that same offset direction; for
        // a true SDF (positive outside, which this field is - see
        // SampleSignedDistance's comment) this converges to the same
        // outward-pointing gradient a 6-tap central difference would,
        // for 1/3 fewer field samples. Was flagged as an immediate
        // optimization target (6x the cost of a single distance query,
        // per contact) - this is that fix, not a "later" one.
        // Deliberately NOT a tight, sub-voxel epsilon (an earlier version
        // used voxelSize*0.1) - a reformed surface is built from many
        // overlapping particle-spheres blended with SmoothMin, which
        // leaves real sub-voxel/few-voxel-scale bumps and seams in the
        // field, not a perfectly smooth analytic surface like Sphere/Box/
        // Capsule ever have. A tight epsilon faithfully resolves every
        // one of those little bumps into the sampled gradient, so a
        // resting sphere's contact normal visibly shifts as its contact
        // point migrates by a fraction of a voxel step to step - the
        // reported jitter on a re-hardened surface that a perfectly
        // analytic collider never shows. A full-voxel epsilon acts as a
        // low-pass filter: it averages across exactly that scale of
        // noise while still tracking genuine curvature at the scale
        // objects in this scene actually rest at.
        const Volume& volume = GetVolume(handle);
        float epsilon = glm::max(volume.desc.voxelSize, 1e-4f);

        const glm::vec3 kD0{ 1.0f, -1.0f, -1.0f };
        const glm::vec3 kD1{ -1.0f, -1.0f, 1.0f };
        const glm::vec3 kD2{ -1.0f, 1.0f, -1.0f };
        const glm::vec3 kD3{ 1.0f, 1.0f, 1.0f };

        glm::vec3 gradient =
            kD0 * SampleSignedDistance(handle, worldPos + kD0 * epsilon) +
            kD1 * SampleSignedDistance(handle, worldPos + kD1 * epsilon) +
            kD2 * SampleSignedDistance(handle, worldPos + kD2 * epsilon) +
            kD3 * SampleSignedDistance(handle, worldPos + kD3 * epsilon);

        float length = glm::length(gradient);
        return length > 0.0f ? gradient / length : glm::vec3(0.0f, 1.0f, 0.0f);
    }

    VoxelSystemGPU::VolumeBounds VoxelSystemGPU::GetVolumeBounds(VoxelVolumeHandle handle) const
    {
        const Volume& volume = GetVolume(handle);
        VolumeBounds bounds;
        bounds.worldMin = volume.desc.origin;
        bounds.worldMax = volume.desc.origin + glm::vec3(volume.desc.VoxelDims()) * volume.desc.voxelSize;
        bounds.voxelSize = volume.desc.voxelSize;
        return bounds;
    }

    void VoxelSystemGPU::SetVolumeNeighbors(VoxelVolumeHandle handle, VoxelVolumeHandle negX, VoxelVolumeHandle posX,
                                             VoxelVolumeHandle negZ, VoxelVolumeHandle posZ)
    {
        Volume& volume = GetVolume(handle);
        volume.neighborNegX = negX;
        volume.neighborPosX = posX;
        volume.neighborNegZ = negZ;
        volume.neighborPosZ = posZ;
    }

    void VoxelSystemGPU::ClearVolume(VoxelVolumeHandle handle)
    {
        Volume& volume = GetVolume(handle);

        // +1.0f, not just kVoxelIsoLevel - a sample sitting exactly AT
        // the iso level is ambiguous/degenerate for marching cubes, so
        // land solidly on the "empty" side, matching the margin
        // SeedSphere's own sphere-SDF fill naturally has away from its
        // isosurface almost everywhere.
        std::fill(volume.densityCPU.begin(), volume.densityCPU.end(), kVoxelIsoLevel + 1.0f);
        UploadDensity(volume);

        volume.dirtyChunkIndices.resize(volume.desc.NumChunks());
        for (uint32_t i = 0; i < volume.desc.NumChunks(); ++i)
        {
            volume.dirtyChunkIndices[i] = i;
        }

        // Every chunk is now genuinely empty - unlike Carve() (which
        // only marks the specific chunks its sphere touched), a full
        // clear means the WHOLE volume's chunkSolid flag should read
        // "not solid" from here on.
        std::fill(volume.chunkSolid.begin(), volume.chunkSolid.end(), false);
    }

    bool VoxelSystemGPU::RaycastSurfaceSegments(VoxelVolumeHandle handle, const glm::vec3& rayOrigin, const glm::vec3& rayDir,
                                                 float maxDistance, std::vector<std::pair<glm::vec3, glm::vec3>>& outSegments) const
    {
        const Volume& volume = GetVolume(handle);
        outSegments.clear();

        glm::vec3 dir = glm::normalize(rayDir);
        glm::vec3 localOrigin = rayOrigin - volume.desc.origin;

        // Half a voxel per step - fine enough to not step over thin
        // carved features, coarse enough that a ~2 unit search range is
        // a few dozen trilinear samples, trivially cheap on CPU. Not
        // true sphere-tracing (stepping by the actual density value,
        // which would be fewer steps in open space) - a fixed step is
        // simpler and safer here since repeated CSG carving doesn't
        // keep the field perfectly metric (exact distances start
        // drifting slightly after several overlapping subtractions),
        // so trusting the density value AS a safe step size the way
        // true sphere tracing does is a little less reliable than it
        // would be for a pristine, un-carved SDF.
        float step = volume.desc.voxelSize * 0.5f;
        constexpr float kIso = kVoxelIsoLevel;
        constexpr int kMaxSegments = 8; // sanity cap, not a expected-case limit

        float prevT = 0.0f;
        float prevDensity = SampleDensityTrilinear(volume, localOrigin);
        bool inSolid = prevDensity < kIso;
        float segmentStartT = inSolid ? 0.0f : 0.0f; // set for real when a crossing is found below

        for (float t = step; t <= maxDistance; t += step)
        {
            glm::vec3 localPos = localOrigin + dir * t;
            float d = SampleDensityTrilinear(volume, localPos);
            bool solidNow = d < kIso;

            if (solidNow != inSolid)
            {
                // Linearly interpolate between (prevT, prevDensity) and
                // (t, d) for the sub-step-accurate point where density
                // crosses the isosurface, rather than snapping to
                // whichever fixed step happened to land past it.
                float denom = (d - prevDensity);
                float frac = (std::abs(denom) > 1e-6f) ? (kIso - prevDensity) / denom : 0.5f;
                float crossingT = prevT + frac * (t - prevT);

                if (!inSolid && solidNow)
                {
                    segmentStartT = crossingT;
                }
                else // was solid, now empty - a full segment just closed
                {
                    outSegments.emplace_back(rayOrigin + dir * segmentStartT, rayOrigin + dir * crossingT);
                    if (static_cast<int>(outSegments.size()) >= kMaxSegments)
                    {
                        return true;
                    }
                }
                inSolid = solidNow;
            }

            prevT = t;
            prevDensity = d;
        }

        if (inSolid)
        {
            // Ray was still inside solid material when the search range
            // ran out - close the segment at maxDistance rather than
            // dropping it (e.g. camera very close to/inside the volume).
            outSegments.emplace_back(rayOrigin + dir * segmentStartT, rayOrigin + dir * maxDistance);
        }

        return !outSegments.empty();
    }
}
