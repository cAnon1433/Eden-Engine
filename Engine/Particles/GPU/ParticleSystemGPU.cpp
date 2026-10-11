#include "ParticleSystemGPU.h"
#include "../../ECS/Components/TransformComponent.h"
#include "../../Physics/ColliderComponent.h"
#include "../../Physics/SDF.h"
#include "../../Voxel/VoxelSystemGPU.h"

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace Eden
{
    namespace
    {
        constexpr uint32_t kLocalSizeX = 256;

        uint32_t DispatchGroupCount(uint32_t particleCount)
        {
            return (particleCount + kLocalSizeX - 1) / kLocalSizeX;
        }

        // Generic full memory barrier - correctness-first, same
        // reasoning VulkanBuffer::InitDeviceLocalWithData gives for
        // vkQueueWaitIdle over a fence: simple and obviously correct
        // beats fine-grained per-buffer barriers for a solver whose
        // compute passes are already sequentially dependent on each
        // other's full output anyway (Density reads what BuildGrid
        // wrote for EVERY particle, not some independently-trackable
        // subset).
        void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                     VkAccessFlags srcAccess, VkAccessFlags dstAccess)
        {
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = srcAccess;
            barrier.dstAccessMask = dstAccess;
            vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        int ColliderShapeToInt(ColliderShape shape)
        {
            switch (shape)
            {
                case ColliderShape::Sphere:  return 0;
                case ColliderShape::Box:     return 1;
                case ColliderShape::Capsule: return 2;
                case ColliderShape::Plane:   return 3;
                case ColliderShape::Voxel:   return 4;
            }
            return 0; // unreachable
        }
    }

    void ParticleSystemGPU::Init(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                                  VkCommandPool commandPool, VkQueue graphicsQueue, VkBuffer voxelDensityBuffer)
    {
        m_Device = device;
        m_Allocator = allocator;
        m_CommandPool = commandPool;
        m_GraphicsQueue = graphicsQueue;

        // Best-effort sanity check, not a guarantee it's checking the
        // exact queue family graphicsQueue came from (this class isn't
        // handed that index, only the queue handle) - see class comment
        // on why compute dispatches deliberately share the graphics
        // queue/command buffer rather than using a second queue family.
        // Fails loudly at Init() instead of producing a confusing
        // vkQueueSubmit error mid-frame if this project is ever ported
        // to hardware where that assumption doesn't hold.
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

        bool anyGraphicsComputeFamily = false;
        for (const auto& family : families)
        {
            if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (family.queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                anyGraphicsComputeFamily = true;
                break;
            }
        }
        if (!anyGraphicsComputeFamily)
        {
            throw std::runtime_error(
                "Eden: ParticleSystemGPU requires a queue family supporting both graphics and "
                "compute - none found on this physical device");
        }

        CreateStorageBuffers();
        CreateComputeDescriptors(voxelDensityBuffer);
        CreateComputePipelines();

        // Push ChemicalTable's default-seeded values (see that field's
        // comment - index 0 reproduces old single-material behavior,
        // index 1 is a ready-to-use second example chemical) to the GPU
        // now that m_ChemicalProperties actually exists. Whoever edits
        // ChemicalTable afterward (code or ImGui) is responsible for
        // calling UploadChemicalTable() again - see that function's
        // comment.
        UploadChemicalTable();
    }

    void ParticleSystemGPU::CreateStorageBuffers()
    {
        // TRANSFER_SRC_BIT included alongside TRANSFER_DST_BIT/
        // STORAGE_BUFFER_BIT - this was actually MISSING before the
        // Chemical ID base build and is a real, separate bug this
        // session's testing surfaced (not something introduced by that
        // work): DownloadRange (used by ReadBackAll/ReadBackColors/
        // KillParticles) does a staged vkCmdCopyBuffer FROM these
        // buffers, and per the Vulkan spec that requires the SOURCE
        // buffer to have TRANSFER_SRC_BIT regardless of what memory type
        // it actually ends up allocated in - a buffer created without it
        // is invalid to use as a copy source even if some driver
        // happens to tolerate it silently. Positions/velocities/heat
        // apparently got away with this before (whatever memory type
        // VMA happened to give them on a given run, or simply not
        // hitting the validation layer's duplicate-message cap), but
        // it's the same latent bug either way, not something specific
        // to the newer buffers below that finally made it visible.
        constexpr VkBufferUsageFlags kParticleUsage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        m_Positions.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_Velocities.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Density/pressure/force are pure scratch - fully overwritten
        // for every active particle index every substep (see
        // particle_density.comp / particle_force.comp), never read
        // before being written this run, so unlike positions/velocities
        // they never need a CPU-side initial upload - TRANSFER_DST is
        // dropped for these three.
        m_Densities.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_Pressures.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_Forces.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Unlike density/pressure/force, heat is NOT fully overwritten
        // for every active index every substep - particle_integrate.comp
        // only touches heat[i] when heat[i] >= 0.0, and skips it
        // entirely otherwise. That means every particle's heat MUST be
        // explicitly initialized at emit time (see Emit/EmitBox/
        // EmitPoints below) rather than left as whatever device memory
        // happened to contain, so this needs kParticleUsage
        // (TRANSFER_DST) like positions/velocities, not the plain
        // STORAGE_BUFFER-only usage the other three scratch buffers get.
        m_Heats.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(float), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Chemical ID system (base build) - see ChemicalTable's comment
        // and m_ChemicalIds/m_Colors/m_ColorsAvg's own comments in the
        // header for which category (identity-data-needing-initial-
        // upload vs. pure-scratch) each falls into.
        m_ChemicalIds.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(uint32_t), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_Colors.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_ColorsAvg.Init(m_Allocator, GPU_MAX_PARTICLES * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        // Small fixed-size table (GPU_MAX_CHEMICALS entries, NOT indexed
        // by particle slot) - needs TRANSFER_DST since UploadChemicalTable
        // writes it via the same staged UploadRange path as everything
        // else here, just infrequently (on Init, and whenever the table
        // is edited) rather than at emit time.
        m_ChemicalProperties.Init(m_Allocator, GPU_MAX_CHEMICALS * sizeof(ChemicalPropertiesGPU), kParticleUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // TRANSFER_DST needed here too - not for CPU upload, but because
        // RecordPendingSteps clears this buffer every substep via
        // vkCmdFillBuffer (a transfer command) rather than a dedicated
        // clear compute shader, which would just be a slower way to
        // write the same zeroes.
        m_CellCounts.Init(m_Allocator, GPU_HASH_TABLE_SIZE * sizeof(uint32_t),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m_CellBuckets.Init(m_Allocator, GPU_HASH_TABLE_SIZE * GPU_MAX_PER_CELL * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Host-visible + persistently mapped, one per pending-step slot -
        // see the class comment on m_ColliderBuffers for why this is an
        // array instead of one shared buffer. Small (GPU_MAX_COLLIDERS
        // entries each), so keeping GPU_MAX_PENDING_STEPS of them mapped
        // simultaneously is negligible memory.
        for (uint32_t i = 0; i < GPU_MAX_PENDING_STEPS; ++i)
        {
            m_ColliderBuffers[i].Init(
                m_Allocator, GPU_MAX_COLLIDERS * sizeof(ColliderGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                &m_ColliderBuffersMapped[i]);
        }
    }

    void ParticleSystemGPU::CreateComputeDescriptors(VkBuffer voxelDensityBuffer)
    {
        // 10, not 9 - binding 9 is VoxelSystemGPU's shared density
        // buffer, added for particle-vs-voxel collision (see this
        // function's voxelDensityBuffer parameter and
        // particle_integrate.comp's DensityBuffer binding). Every other
        // shader in this set (build_grid/density/force) simply never
        // declares binding 9 in its own GLSL - same "reserved but not
        // every shader touches every binding" pattern bindings 3/5/6
        // already have.
        // 10, not 9 - binding 9 is VoxelSystemGPU's shared density
        // buffer, added for particle-vs-voxel collision (see this
        // function's voxelDensityBuffer parameter and
        // particle_integrate.comp's DensityBuffer binding). Every other
        // shader in this set (build_grid/density/force) simply never
        // declares binding 9 in its own GLSL - same "reserved but not
        // every shader touches every binding" pattern bindings 3/5/6
        // already have.
        // 14, not 10 - bindings 10-13 are the Chemical ID system's
        // buffers (chemicalIds, colors, colorsAvg, chemicalProperties -
        // see those members' comments in ParticleSystemGPU.h).
        // particle_build_grid.comp doesn't declare any of bindings
        // 10-13 (it never needs per-particle chemical data, only
        // positions), same "reserved but not every shader touches every
        // binding" pattern bindings 3/5/6/9 already established.
        std::array<VkDescriptorSetLayoutBinding, 14> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ComputeSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: failed to create ParticleSystemGPU compute descriptor set layout");
        }

        std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, GPU_MAX_PENDING_STEPS * static_cast<uint32_t>(bindings.size()) },
        };
        m_ComputeDescriptorPool.Init(m_Device, poolSizes, GPU_MAX_PENDING_STEPS);

        for (uint32_t slot = 0; slot < GPU_MAX_PENDING_STEPS; ++slot)
        {
            VkDescriptorSet set = m_ComputeDescriptorPool.AllocateSet(m_ComputeSetLayout);
            m_ComputeDescriptorSets[slot] = set;

            std::array<VkDescriptorBufferInfo, 14> bufferInfos{};
            bufferInfos[0] = { m_Positions.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[1] = { m_Velocities.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[2] = { m_Densities.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[3] = { m_Pressures.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[4] = { m_Forces.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[5] = { m_CellCounts.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[6] = { m_CellBuckets.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[7] = { m_ColliderBuffers[slot].Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[8] = { m_Heats.Get(), 0, VK_WHOLE_SIZE };
            // voxelDensityBuffer may legitimately be VK_NULL_HANDLE if
            // this project ever calls Init() before VoxelSystemGPU has
            // registered a first volume - VoxelSystemGPU::Init() itself
            // always creates m_SharedDensityBuffer unconditionally
            // though (see that function), so in practice this is always
            // a real handle by the time main.cpp's Init ordering reaches
            // here (voxelSystem.Init() runs first - see this class's
            // Init() header comment). Bound as-is either way; Vulkan
            // permits a VK_NULL_HANDLE descriptor write to go
            // unexercised as long as nothing actually reads through it
            // (ShapeDistance's Voxel branch is only reached for
            // scenes that actually have a Voxel collider, which
            // wouldn't exist yet if VoxelSystemGPU were skipped
            // entirely).
            bufferInfos[9] = { voxelDensityBuffer, 0, VK_WHOLE_SIZE };
            bufferInfos[10] = { m_ChemicalIds.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[11] = { m_Colors.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[12] = { m_ColorsAvg.Get(), 0, VK_WHOLE_SIZE };
            bufferInfos[13] = { m_ChemicalProperties.Get(), 0, VK_WHOLE_SIZE };

            std::array<VkWriteDescriptorSet, 14> writes{};
            for (uint32_t b = 0; b < writes.size(); ++b)
            {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = set;
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &bufferInfos[b];
            }

            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    void ParticleSystemGPU::CreateComputePipelines()
    {
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SimParamsGPU);

        m_ComputePipelineLayout.Init(m_Device, { m_ComputeSetLayout }, { pushConstantRange });

        m_BuildGridPipeline.Init(m_Device, m_ComputePipelineLayout.Get(), "Shaders/Compiled/particle_build_grid.comp.spv");
        m_DensityPipeline.Init(m_Device, m_ComputePipelineLayout.Get(), "Shaders/Compiled/particle_density.comp.spv");
        m_ForcePipeline.Init(m_Device, m_ComputePipelineLayout.Get(), "Shaders/Compiled/particle_force.comp.spv");
        m_IntegratePipeline.Init(m_Device, m_ComputePipelineLayout.Get(), "Shaders/Compiled/particle_integrate.comp.spv");
    }

    void ParticleSystemGPU::Shutdown()
    {
        if (m_UploadFence != VK_NULL_HANDLE)
        {
            vkDestroyFence(m_Device, m_UploadFence, nullptr);
            m_UploadFence = VK_NULL_HANDLE;
        }

        if (m_UploadCommandBuffer != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &m_UploadCommandBuffer);
            m_UploadCommandBuffer = VK_NULL_HANDLE;
        }

        if (m_UploadStagingBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_Allocator, m_UploadStagingBuffer, m_UploadStagingAllocation);
            m_UploadStagingBuffer = VK_NULL_HANDLE;
            m_UploadStagingAllocation = VK_NULL_HANDLE;
            m_UploadStagingMappedPtr = nullptr;
            m_UploadStagingCapacityBytes = 0;
        }

        m_BuildGridPipeline.Shutdown();
        m_DensityPipeline.Shutdown();
        m_ForcePipeline.Shutdown();
        m_IntegratePipeline.Shutdown();
        m_ComputePipelineLayout.Shutdown();
        m_ComputeDescriptorPool.Shutdown();

        if (m_ComputeSetLayout != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(m_Device, m_ComputeSetLayout, nullptr);
            m_ComputeSetLayout = VK_NULL_HANDLE;
        }

        for (auto& buffer : m_ColliderBuffers)
        {
            buffer.Shutdown();
        }
        m_CellBuckets.Shutdown();
        m_CellCounts.Shutdown();
        m_Heats.Shutdown();
        m_ChemicalIds.Shutdown();
        m_Colors.Shutdown();
        m_ColorsAvg.Shutdown();
        m_ChemicalProperties.Shutdown();
        m_Forces.Shutdown();
        m_Pressures.Shutdown();
        m_Densities.Shutdown();
        m_Velocities.Shutdown();
        m_Positions.Shutdown();
    }

    void ParticleSystemGPU::EnsureUploadStagingCapacity(VkDeviceSize requiredBytes)
    {
        if (requiredBytes <= m_UploadStagingCapacityBytes)
        {
            return; // common case - existing buffer already has enough room, zero allocation this call
        }

        if (m_UploadStagingBuffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_Allocator, m_UploadStagingBuffer, m_UploadStagingAllocation);
            m_UploadStagingBuffer = VK_NULL_HANDLE;
            m_UploadStagingAllocation = VK_NULL_HANDLE;
            m_UploadStagingMappedPtr = nullptr;
            m_UploadStagingCapacityBytes = 0;
        }

        // Grow with headroom (doubling), not exactly-to-fit - see
        // VoxelSystemGPU::EnsureDensityStagingCapacity's identical
        // reasoning.
        VkDeviceSize newCapacity = std::max<VkDeviceSize>(requiredBytes, m_UploadStagingCapacityBytes * 2);

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = newCapacity;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &m_UploadStagingBuffer,
                             &m_UploadStagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: ParticleSystemGPU failed to (re)create the persistent upload staging buffer");
        }

        m_UploadStagingMappedPtr = stagingInfoOut.pMappedData;
        m_UploadStagingCapacityBytes = newCapacity;
    }

    VkCommandBuffer ParticleSystemGPU::GetReusableUploadCommandBuffer()
    {
        if (m_UploadCommandBuffer == VK_NULL_HANDLE)
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandPool = m_CommandPool;
            cmdAllocInfo.commandBufferCount = 1;
            vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &m_UploadCommandBuffer);
        }
        else
        {
            vkResetCommandBuffer(m_UploadCommandBuffer, 0);
        }
        return m_UploadCommandBuffer;
    }

    void ParticleSystemGPU::SubmitAndWaitForUploadCommandBuffer()
    {
        if (m_UploadFence == VK_NULL_HANDLE)
        {
            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(m_Device, &fenceInfo, nullptr, &m_UploadFence) != VK_SUCCESS)
            {
                m_UploadFence = VK_NULL_HANDLE;
            }
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_UploadCommandBuffer;

        if (m_UploadFence == VK_NULL_HANDLE)
        {
            // Fallback matching VoxelSystemGPU::SubmitAndWaitForCommandBuffer's
            // own fallback - no working fence, so fall back to the
            // always-correct-if-slower vkQueueWaitIdle rather than
            // proceeding with no wait at all.
            vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_GraphicsQueue);
            return;
        }

        vkResetFences(m_Device, 1, &m_UploadFence);
        vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, m_UploadFence);
        vkWaitForFences(m_Device, 1, &m_UploadFence, VK_TRUE, UINT64_MAX);
    }

    void ParticleSystemGPU::UploadRange(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size)
    {
        EnsureUploadStagingCapacity(size);
        std::memcpy(m_UploadStagingMappedPtr, data, static_cast<size_t>(size));

        VkCommandBuffer cmd = GetReusableUploadCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, dst, 1, &copyRegion);

        vkEndCommandBuffer(cmd);
        SubmitAndWaitForUploadCommandBuffer();
    }

    void ParticleSystemGPU::DownloadRange(VkBuffer src, VkDeviceSize srcOffset, void* data, VkDeviceSize size) const
    {
        if (size == 0)
        {
            return;
        }

        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;

        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        // RANDOM, not SEQUENTIAL_WRITE like UploadRange's staging buffer
        // - this one is read from the CPU side after the copy, the
        // opposite access pattern.
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfoOut{};
        if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingInfoOut) != VK_SUCCESS)
        {
            throw std::runtime_error("Eden: ParticleSystemGPU failed to create readback staging buffer");
        }

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

        // Unlike UploadRange (writing into a buffer nothing else reads
        // until the next dispatch), `src` here is live simulation state
        // - the last compute pass that touched it (particle_integrate.comp,
        // almost always) needs to finish before this copy reads it.
        VkMemoryBarrier preCopyBarrier{};
        preCopyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        preCopyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        preCopyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 1, &preCopyBarrier, 0, nullptr, 0, nullptr);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.size = size;
        vkCmdCopyBuffer(cmd, src, stagingBuffer, 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_GraphicsQueue);

        std::memcpy(data, stagingInfoOut.pMappedData, static_cast<size_t>(size));

        vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &cmd);
        vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
    }

    void ParticleSystemGPU::ReadBackAll(std::vector<glm::vec4>& outPositions, std::vector<glm::vec4>& outVelocities,
                                         std::vector<float>& outHeat, std::vector<uint32_t>& outChemicalIds) const
    {
        outPositions.resize(m_ParticleCount);
        outVelocities.resize(m_ParticleCount);
        outHeat.resize(m_ParticleCount);
        outChemicalIds.resize(m_ParticleCount);

        if (m_ParticleCount == 0)
        {
            return;
        }

        DownloadRange(m_Positions.Get(), 0, outPositions.data(), m_ParticleCount * sizeof(glm::vec4));
        DownloadRange(m_Velocities.Get(), 0, outVelocities.data(), m_ParticleCount * sizeof(glm::vec4));
        DownloadRange(m_Heats.Get(), 0, outHeat.data(), m_ParticleCount * sizeof(float));
        DownloadRange(m_ChemicalIds.Get(), 0, outChemicalIds.data(), m_ParticleCount * sizeof(uint32_t));
    }

    void ParticleSystemGPU::ReadBackColors(std::vector<glm::vec4>& outColors) const
    {
        outColors.resize(m_ParticleCount);
        if (m_ParticleCount == 0)
        {
            return;
        }
        DownloadRange(m_Colors.Get(), 0, outColors.data(), m_ParticleCount * sizeof(glm::vec4));
    }

    void ParticleSystemGPU::KillParticles(const std::vector<uint32_t>& indices)
    {
        if (indices.empty() || m_ParticleCount == 0)
        {
            return;
        }

        std::vector<glm::vec4> positions, velocities;
        std::vector<float> heats;
        std::vector<uint32_t> chemicalIds;
        ReadBackAll(positions, velocities, heats, chemicalIds);

        // Colors aren't part of ReadBackAll (nothing outside this
        // function needs them today - see that function's own comment on
        // why it only reads back what ReformSystem/ClusterSystem
        // actually use), but they DO need the same compact-on-removal
        // treatment as positions/heat/chemicalIds - it's per-particle
        // identity data, not scratch (see m_Colors' header comment) - so
        // a separate, dedicated download here rather than growing
        // ReadBackAll's signature for a field only this function needs.
        std::vector<glm::vec4> colors(m_ParticleCount);
        DownloadRange(m_Colors.Get(), 0, colors.data(), m_ParticleCount * sizeof(glm::vec4));

        std::vector<bool> remove(m_ParticleCount, false);
        for (uint32_t index : indices)
        {
            if (index < m_ParticleCount)
            {
                remove[index] = true;
            }
        }

        std::vector<glm::vec4> keptPositions, keptVelocities, keptColors;
        std::vector<float> keptHeats;
        std::vector<uint32_t> keptChemicalIds;
        keptPositions.reserve(m_ParticleCount);
        keptVelocities.reserve(m_ParticleCount);
        keptHeats.reserve(m_ParticleCount);
        keptChemicalIds.reserve(m_ParticleCount);
        keptColors.reserve(m_ParticleCount);

        for (uint32_t i = 0; i < m_ParticleCount; ++i)
        {
            if (!remove[i])
            {
                keptPositions.push_back(positions[i]);
                keptVelocities.push_back(velocities[i]);
                keptHeats.push_back(heats[i]);
                keptChemicalIds.push_back(chemicalIds[i]);
                keptColors.push_back(colors[i]);
            }
        }

        uint32_t newCount = static_cast<uint32_t>(keptPositions.size());
        if (newCount > 0)
        {
            UploadRange(m_Positions.Get(), 0, keptPositions.data(), newCount * sizeof(glm::vec4));
            UploadRange(m_Velocities.Get(), 0, keptVelocities.data(), newCount * sizeof(glm::vec4));
            UploadRange(m_Heats.Get(), 0, keptHeats.data(), newCount * sizeof(float));
            UploadRange(m_ChemicalIds.Get(), 0, keptChemicalIds.data(), newCount * sizeof(uint32_t));
            UploadRange(m_Colors.Get(), 0, keptColors.data(), newCount * sizeof(glm::vec4));
        }
        m_ParticleCount = newCount;
    }

    void ParticleSystemGPU::UploadChemicalTable()
    {
        UploadRange(m_ChemicalProperties.Get(), 0, ChemicalTable.data(), ChemicalTable.size() * sizeof(ChemicalPropertiesGPU));
    }

    void ParticleSystemGPU::Emit(const glm::vec3& position, const glm::vec3& velocity, uint32_t chemicalId)
    {
        if (m_ParticleCount >= GPU_MAX_PARTICLES)
        {
            std::cerr << "Eden: ParticleSystemGPU at capacity (" << GPU_MAX_PARTICLES
                      << "), dropping Emit() call - see GPU_MAX_PARTICLES in ParticleGPUTypes.h\n";
            return;
        }

        uint32_t clampedChemicalId = std::min(chemicalId, GPU_MAX_CHEMICALS - 1);
        glm::vec4 pos4(position, 1.0f);
        glm::vec4 vel4(velocity, 0.0f);
        float heat4 = 1.0f; // starts warm, needs to cool - see GPU_HEAT_NOT_MELTED's comment
        glm::vec4 color4 = ChemicalTable[clampedChemicalId].baseColor;
        UploadRange(m_Positions.Get(), m_ParticleCount * sizeof(glm::vec4), &pos4, sizeof(glm::vec4));
        UploadRange(m_Velocities.Get(), m_ParticleCount * sizeof(glm::vec4), &vel4, sizeof(glm::vec4));
        UploadRange(m_Heats.Get(), m_ParticleCount * sizeof(float), &heat4, sizeof(float));
        UploadRange(m_ChemicalIds.Get(), m_ParticleCount * sizeof(uint32_t), &clampedChemicalId, sizeof(uint32_t));
        UploadRange(m_Colors.Get(), m_ParticleCount * sizeof(glm::vec4), &color4, sizeof(glm::vec4));
        ++m_ParticleCount;
    }

    void ParticleSystemGPU::EmitBox(const glm::vec3& min, const glm::vec3& max, float spacing,
                                     const glm::vec3& initialVelocity, uint32_t chemicalId)
    {
        // Same fill-spacing convention as ParticleSystem::EmitBox - see
        // its ParticleSystem.h comment for why 0.6 * h.
        float actualSpacing = spacing > 0.0f ? spacing : smoothingRadius * 0.6f;
        uint32_t clampedChemicalId = std::min(chemicalId, GPU_MAX_CHEMICALS - 1);
        glm::vec4 color4 = ChemicalTable[clampedChemicalId].baseColor;

        std::vector<glm::vec4> positions;
        std::vector<glm::vec4> velocities;
        std::vector<float> heats;
        std::vector<uint32_t> chemicalIds;
        std::vector<glm::vec4> colors;

        for (float x = min.x; x <= max.x; x += actualSpacing)
        {
            for (float y = min.y; y <= max.y; y += actualSpacing)
            {
                for (float z = min.z; z <= max.z; z += actualSpacing)
                {
                    positions.emplace_back(x, y, z, 1.0f);
                    velocities.emplace_back(initialVelocity, 0.0f);
                    heats.push_back(1.0f); // starts warm, needs to cool - see GPU_HEAT_NOT_MELTED's comment
                    chemicalIds.push_back(clampedChemicalId);
                    colors.push_back(color4);
                }
            }
        }

        uint32_t availableCapacity = GPU_MAX_PARTICLES - m_ParticleCount;
        uint32_t countToUpload = std::min(static_cast<uint32_t>(positions.size()), availableCapacity);

        if (countToUpload < positions.size())
        {
            std::cerr << "Eden: ParticleSystemGPU::EmitBox requested " << positions.size()
                      << " particles but only " << availableCapacity << " of capacity " << GPU_MAX_PARTICLES
                      << " remain - truncating (see GPU_MAX_PARTICLES in ParticleGPUTypes.h)\n";
        }

        if (countToUpload == 0)
        {
            return;
        }

        // Batched into ONE staging write + ONE command buffer (five
        // copy regions, one per destination buffer - vkCmdCopyBuffer
        // can't target multiple different destinations in a single call,
        // but multiple vkCmdCopyBuffer calls recorded into the same
        // command buffer still submit and wait exactly once) + ONE
        // submit+wait, instead of five separate UploadRange calls. Even
        // with UploadRange's own persistent resources (see that
        // function's comment), five separate submissions per EmitBox
        // call is still five separate driver round trips - with
        // ProcessPourInput (main.cpp) calling EmitBox on a held key at
        // ~12/sec, that's up to 60 submissions/sec for something that
        // only needs 1. This is what actually gets that down to 12/sec.
        VkDeviceSize positionsBytes = static_cast<VkDeviceSize>(countToUpload) * sizeof(glm::vec4);
        VkDeviceSize velocitiesBytes = static_cast<VkDeviceSize>(countToUpload) * sizeof(glm::vec4);
        VkDeviceSize heatsBytes = static_cast<VkDeviceSize>(countToUpload) * sizeof(float);
        VkDeviceSize chemicalIdsBytes = static_cast<VkDeviceSize>(countToUpload) * sizeof(uint32_t);
        VkDeviceSize colorsBytes = static_cast<VkDeviceSize>(countToUpload) * sizeof(glm::vec4);
        VkDeviceSize totalBytes = positionsBytes + velocitiesBytes + heatsBytes + chemicalIdsBytes + colorsBytes;

        EnsureUploadStagingCapacity(totalBytes);

        char* stagingPtr = static_cast<char*>(m_UploadStagingMappedPtr);
        VkDeviceSize positionsStagingOffset = 0;
        VkDeviceSize velocitiesStagingOffset = positionsStagingOffset + positionsBytes;
        VkDeviceSize heatsStagingOffset = velocitiesStagingOffset + velocitiesBytes;
        VkDeviceSize chemicalIdsStagingOffset = heatsStagingOffset + heatsBytes;
        VkDeviceSize colorsStagingOffset = chemicalIdsStagingOffset + chemicalIdsBytes;

        std::memcpy(stagingPtr + positionsStagingOffset, positions.data(), static_cast<size_t>(positionsBytes));
        std::memcpy(stagingPtr + velocitiesStagingOffset, velocities.data(), static_cast<size_t>(velocitiesBytes));
        std::memcpy(stagingPtr + heatsStagingOffset, heats.data(), static_cast<size_t>(heatsBytes));
        std::memcpy(stagingPtr + chemicalIdsStagingOffset, chemicalIds.data(), static_cast<size_t>(chemicalIdsBytes));
        std::memcpy(stagingPtr + colorsStagingOffset, colors.data(), static_cast<size_t>(colorsBytes));

        VkCommandBuffer cmd = GetReusableUploadCommandBuffer();

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy positionsCopy{};
        positionsCopy.srcOffset = positionsStagingOffset;
        positionsCopy.dstOffset = m_ParticleCount * sizeof(glm::vec4);
        positionsCopy.size = positionsBytes;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, m_Positions.Get(), 1, &positionsCopy);

        VkBufferCopy velocitiesCopy{};
        velocitiesCopy.srcOffset = velocitiesStagingOffset;
        velocitiesCopy.dstOffset = m_ParticleCount * sizeof(glm::vec4);
        velocitiesCopy.size = velocitiesBytes;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, m_Velocities.Get(), 1, &velocitiesCopy);

        VkBufferCopy heatsCopy{};
        heatsCopy.srcOffset = heatsStagingOffset;
        heatsCopy.dstOffset = m_ParticleCount * sizeof(float);
        heatsCopy.size = heatsBytes;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, m_Heats.Get(), 1, &heatsCopy);

        VkBufferCopy chemicalIdsCopy{};
        chemicalIdsCopy.srcOffset = chemicalIdsStagingOffset;
        chemicalIdsCopy.dstOffset = m_ParticleCount * sizeof(uint32_t);
        chemicalIdsCopy.size = chemicalIdsBytes;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, m_ChemicalIds.Get(), 1, &chemicalIdsCopy);

        VkBufferCopy colorsCopy{};
        colorsCopy.srcOffset = colorsStagingOffset;
        colorsCopy.dstOffset = m_ParticleCount * sizeof(glm::vec4);
        colorsCopy.size = colorsBytes;
        vkCmdCopyBuffer(cmd, m_UploadStagingBuffer, m_Colors.Get(), 1, &colorsCopy);

        vkEndCommandBuffer(cmd);
        SubmitAndWaitForUploadCommandBuffer();

        m_ParticleCount += countToUpload;
    }

    void ParticleSystemGPU::EmitPoints(const std::vector<glm::vec3>& positions, const glm::vec3& initialVelocity,
                                        float initialHeat, uint32_t chemicalId, const glm::vec4& colorOverride)
    {
        if (positions.empty())
        {
            return;
        }

        uint32_t clampedChemicalId = std::min(chemicalId, GPU_MAX_CHEMICALS - 1);
        // vec4(0,0,0,0) means "no override" - see this function's
        // header comment (reuses VoxelSystemGPU::SetTransform's existing
        // convention for the same meaning).
        bool hasOverride = colorOverride != glm::vec4(0.0f);
        glm::vec4 resolvedColor = hasOverride ? colorOverride : ChemicalTable[clampedChemicalId].baseColor;

        uint32_t availableCapacity = GPU_MAX_PARTICLES - m_ParticleCount;
        uint32_t countToUpload = std::min(static_cast<uint32_t>(positions.size()), availableCapacity);

        if (countToUpload < positions.size())
        {
            std::cerr << "Eden: ParticleSystemGPU::EmitPoints requested " << positions.size()
                      << " particles but only " << availableCapacity << " of capacity " << GPU_MAX_PARTICLES
                      << " remain - truncating (see GPU_MAX_PARTICLES in ParticleGPUTypes.h)\n";
        }

        if (countToUpload == 0)
        {
            return;
        }

        std::vector<glm::vec4> uploadPositions(countToUpload);
        std::vector<glm::vec4> uploadVelocities(countToUpload, glm::vec4(initialVelocity, 0.0f));
        std::vector<float> uploadHeats(countToUpload, initialHeat);
        std::vector<uint32_t> uploadChemicalIds(countToUpload, clampedChemicalId);
        std::vector<glm::vec4> uploadColors(countToUpload, resolvedColor);
        for (uint32_t i = 0; i < countToUpload; ++i)
        {
            uploadPositions[i] = glm::vec4(positions[i], 1.0f);
        }

        UploadRange(m_Positions.Get(), m_ParticleCount * sizeof(glm::vec4), uploadPositions.data(), countToUpload * sizeof(glm::vec4));
        UploadRange(m_Velocities.Get(), m_ParticleCount * sizeof(glm::vec4), uploadVelocities.data(), countToUpload * sizeof(glm::vec4));
        UploadRange(m_Heats.Get(), m_ParticleCount * sizeof(float), uploadHeats.data(), countToUpload * sizeof(float));
        UploadRange(m_ChemicalIds.Get(), m_ParticleCount * sizeof(uint32_t), uploadChemicalIds.data(), countToUpload * sizeof(uint32_t));
        UploadRange(m_Colors.Get(), m_ParticleCount * sizeof(glm::vec4), uploadColors.data(), countToUpload * sizeof(glm::vec4));
        m_ParticleCount += countToUpload;
    }

    void ParticleSystemGPU::Clear()
    {
        // Buffer contents left as-is - every pass is bounded by
        // sim.particleCount (see the compute shaders' `if (i >=
        // particleCount) return;` guards), so stale data past the new,
        // lower count is simply never read. A future Emit() after
        // Clear() overwrites from index 0 again.
        m_ParticleCount = 0;
    }

    SimParamsGPU ParticleSystemGPU::BuildSimParams(float subDt, uint32_t numColliders) const
    {
        SimParamsGPU params{};
        params.particleCount = m_ParticleCount;
        params.numColliders = numColliders;
        params.hashTableSize = GPU_HASH_TABLE_SIZE;
        params.maxPerCell = GPU_MAX_PER_CELL;
        params.h = smoothingRadius;
        params.h2 = smoothingRadius * smoothingRadius;
        params.mass = particleMass;
        params.restDensity = restDensity;
        params.stiffness = stiffness;
        params.gamma = gamma;
        params.viscosity = viscosityCoefficient;
        params.dt = subDt;
        params.cellSize = smoothingRadius; // matches the CPU grid's own sizing assumption
        params.particleRadius = boundaryRadius;
        params.boundaryRestitution = boundaryRestitution;
        params.boundaryFriction = boundaryFriction;
        params.maxAcceleration = maxAcceleration;
        params.maxSweepIterations = static_cast<uint32_t>(std::max(1, maxSweepIterations));
        params.heatDecayRate = heatDecayRate;
        params.heatColdMultiplier = heatColdMultiplier;
        params.gravity = glm::vec4(gravity, 0.0f);
        params.voidKillY = voidKillY;
        params.cohesion = cohesion;
        return params;
    }

    void ParticleSystemGPU::Step(Registry& registry, float fixedDeltaTime, const VoxelSystemGPU* voxelSystem)
    {
        if (m_ParticleCount == 0)
        {
            return;
        }

        if (m_PendingSteps.size() >= GPU_MAX_PENDING_STEPS)
        {
            // Shouldn't happen at today's MAX_PHYSICS_STEPS_PER_FRAME
            // (main.cpp) - defensive cap-and-warn, same pattern as
            // Renderer::DrawFrame's MAX_INSTANCES_PER_FRAME check.
            static bool s_Warned = false;
            if (!s_Warned)
            {
                std::cerr << "Eden: ParticleSystemGPU has " << GPU_MAX_PENDING_STEPS
                          << " unflushed Step() calls queued - dropping further Step() calls until "
                             "RecordPendingSteps() runs (see GPU_MAX_PENDING_STEPS in ParticleGPUTypes.h)\n";
                s_Warned = true;
            }
            return;
        }

        // Same snapshot-once-per-Step()-call reasoning as
        // ParticleSystem::ResolveBoundaries - colliders are few, reused
        // across every substep this call queues.
        auto colliderEntities = registry.View<TransformComponent, ColliderComponent>();

        PendingStep pending;
        pending.fixedDeltaTime = fixedDeltaTime;
        pending.colliders.reserve(colliderEntities.size());

        for (Entity entity : colliderEntities)
        {
            if (pending.colliders.size() >= GPU_MAX_COLLIDERS)
            {
                static bool s_WarnedColliders = false;
                if (!s_WarnedColliders)
                {
                    std::cerr << "Eden: ParticleSystemGPU scene has more than " << GPU_MAX_COLLIDERS
                              << " colliders - extras ignored for GPU particle boundary collision "
                                 "(see GPU_MAX_COLLIDERS in ParticleGPUTypes.h)\n";
                    s_WarnedColliders = true;
                }
                break;
            }

            const auto& transform = registry.GetComponent<TransformComponent>(entity);
            const auto& collider = registry.GetComponent<ColliderComponent>(entity);

            ColliderComponent scaled = SDF::ScaledCollider(collider, transform.scale);
            glm::mat3 rotation = SDF::RotationMatrixFromDegrees(transform.rotationDegrees);
            glm::vec3 worldCenter = transform.position + rotation * scaled.localOffset;

            ColliderGPU gpuCollider{};
            gpuCollider.rotRight = glm::vec4(rotation[0], 0.0f);
            gpuCollider.rotUp = glm::vec4(rotation[1], 0.0f);
            gpuCollider.rotForward = glm::vec4(rotation[2], 0.0f);
            gpuCollider.worldCenterType = glm::vec4(worldCenter, static_cast<float>(ColliderShapeToInt(scaled.shape)));

            switch (scaled.shape)
            {
                case ColliderShape::Sphere:  gpuCollider.shapeParams = glm::vec4(scaled.radius, 0.0f, 0.0f, 0.0f); break;
                case ColliderShape::Box:     gpuCollider.shapeParams = glm::vec4(scaled.halfExtents, 0.0f); break;
                case ColliderShape::Capsule: gpuCollider.shapeParams = glm::vec4(scaled.halfHeight, scaled.radius, 0.0f, 0.0f); break;
                case ColliderShape::Plane:   gpuCollider.shapeParams = glm::vec4(scaled.planeNormal, 0.0f); break;
                case ColliderShape::Voxel:
                {
                    // No rotation/worldCenter for Voxel - see
                    // ColliderGPU's struct comment on why
                    // worldCenterType.xyz is repurposed to hold
                    // desc.origin instead for this shape, matching
                    // VoxelSystemGPU::SampleSignedDistance's own
                    // rotation-free `worldPos - origin` convention
                    // exactly (voxel volumes are always Static,
                    // axis-aligned - see ColliderComponent.h). Overwrite
                    // what was just computed above (worldCenter/rotation
                    // are meaningless for this shape and simply
                    // discarded) rather than branching earlier to skip
                    // computing them - they're cheap, and keeping this
                    // switch's shape as "always compute rotation/
                    // worldCenter, then override per-shape as needed"
                    // matches every other case's structure.
                    if (voxelSystem == nullptr || scaled.voxelVolume == InvalidVoxelVolumeHandle)
                    {
                        // No voxel system passed to Step (see this
                        // function's header comment - optional,
                        // matching CollisionSystem::Step's own
                        // contract), or this collider's handle was
                        // never set. Upload a harmless no-op entry
                        // rather than skipping the push_back below -
                        // skipping would shift every SUBSEQUENT
                        // collider's index for this Step() call, which
                        // is exactly the kind of "colliders[c] used to
                        // mean something else" bug ReformSystem's
                        // batched-kill fix (see the to-do doc) was
                        // written to avoid elsewhere in this project.
                        // shapeParams/voxelParams/voxelSampleDims all
                        // stay zeroed; ShapeDistance's Voxel branch
                        // treats sampleDims==(0,0,0) as "always outside,
                        // no contact" (see that branch's own comment).
                        static bool s_WarnedMissingVoxelSystem = false;
                        if (!s_WarnedMissingVoxelSystem)
                        {
                            std::cerr << "Eden: ParticleSystemGPU::Step saw a Voxel collider but no "
                                         "VoxelSystemGPU was passed (or its voxelVolume handle is unset) - "
                                         "that collider will not affect GPU particles this call\n";
                            s_WarnedMissingVoxelSystem = true;
                        }
                        break;
                    }

                    VoxelSystemGPU::VolumeBounds bounds = voxelSystem->GetVolumeBounds(scaled.voxelVolume);
                    gpuCollider.worldCenterType = glm::vec4(bounds.worldMin,
                                                             static_cast<float>(ColliderShapeToInt(scaled.shape)));

                    glm::ivec3 sampleDims = voxelSystem->GetSampleDims(scaled.voxelVolume);
                    // densityOffset stored as a float here (matching
                    // this struct's std430 float-only convention - see
                    // voxelParams' own comment) and reinterpreted back
                    // to uint in ShapeDistance's Voxel branch. Safe up
                    // to 2^24 (~16.7M) without precision loss - see that
                    // branch's comment for why this project's actual
                    // shared-buffer capacity is comfortably under that.
                    float densityOffsetAsFloat =
                        static_cast<float>(voxelSystem->GetDensityBufferOffset(scaled.voxelVolume));
                    gpuCollider.voxelParams = glm::vec4(densityOffsetAsFloat, bounds.voxelSize, 0.0f, 0.0f);
                    gpuCollider.voxelSampleDims = glm::ivec4(sampleDims, 0);
                    break;
                }
            }

            pending.colliders.push_back(gpuCollider);
        }

        m_PendingSteps.push_back(std::move(pending));
    }

    void ParticleSystemGPU::RecordPendingSteps(VkCommandBuffer cmd)
    {
        for (size_t slot = 0; slot < m_PendingSteps.size(); ++slot)
        {
            const PendingStep& step = m_PendingSteps[slot];

            if (!step.colliders.empty())
            {
                std::memcpy(m_ColliderBuffersMapped[slot], step.colliders.data(),
                            step.colliders.size() * sizeof(ColliderGPU));
            }

            float subDt = step.fixedDeltaTime / static_cast<float>(std::max(1, substeps));
            SimParamsGPU params = BuildSimParams(subDt, static_cast<uint32_t>(step.colliders.size()));
            uint32_t groupCount = DispatchGroupCount(m_ParticleCount);

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComputePipelineLayout.Get(),
                                     0, 1, &m_ComputeDescriptorSets[slot], 0, nullptr);
            vkCmdPushConstants(cmd, m_ComputePipelineLayout.Get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SimParamsGPU), &params);

            for (int sub = 0; sub < std::max(1, substeps); ++sub)
            {
                // Clear the grid's atomic counters - a previous
                // substep's Density/Force passes may still be reading
                // the OLD counts right up until this point, hence the
                // barrier before the fill, not just after it.
                Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                vkCmdFillBuffer(cmd, m_CellCounts.Get(), 0, VK_WHOLE_SIZE, 0);
                Barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildGridPipeline.Get());
                vkCmdDispatch(cmd, groupCount, 1, 1);
                Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DensityPipeline.Get());
                vkCmdDispatch(cmd, groupCount, 1, 1);
                Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ForcePipeline.Get());
                vkCmdDispatch(cmd, groupCount, 1, 1);
                Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_IntegratePipeline.Get());
                vkCmdDispatch(cmd, groupCount, 1, 1);

                // Positions/velocities just changed - next substep's
                // BuildGrid (or, on the very last substep of the very
                // last pending step, Renderer::DrawFrame's own
                // compute -> vertex-shader barrier) needs this.
                Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            }
        }

        m_PendingSteps.clear();
    }
}
