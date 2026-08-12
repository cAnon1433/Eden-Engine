# VMA (Vulkan Memory Allocator)

vk_mem_alloc.h is vendored here and IS used, as of the vertex buffer
milestone. VulkanMemoryAllocator.cpp is the one translation unit that
defines VMA_IMPLEMENTATION - don't add that define anywhere else, VMA
will fail to link with duplicate symbol errors if you do.

Everything in Resources/ allocates through Eden::VulkanMemoryAllocator
(a thin wrapper around VmaAllocator) rather than calling
vkAllocateMemory directly.
