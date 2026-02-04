#include "memory_manager.h"

#include <cstdlib>

size_t MemoryManager::total_allocated_ = 0;

/**
 * @brief Initialize memory tracking (stub).
 */
void MemoryManager::init() {
    total_allocated_ = 0;
}

/**
 * @brief Allocate a block of memory (stub).
 */
void* MemoryManager::allocate(size_t size) {
    // NOTE: Using malloc as a placeholder; replace with custom allocator.
    void* ptr = std::malloc(size);
    if (ptr != nullptr) {
        total_allocated_ += size;
    }
    return ptr;
}

/**
 * @brief Deallocate a block of memory (stub).
 */
void MemoryManager::deallocate(void* ptr) {
    // NOTE: Without size info, we cannot decrement total_allocated_ here.
    std::free(ptr);
}

/**
 * @brief Current total allocated bytes (stub).
 */
size_t MemoryManager::totalAllocated() {
    return total_allocated_;
}
