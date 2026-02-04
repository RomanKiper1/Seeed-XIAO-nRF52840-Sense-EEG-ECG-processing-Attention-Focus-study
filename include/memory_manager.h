#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <cstddef>

/**
 * @brief Centralized memory management interface (stub).
 *
 * Responsibility: unify allocation/deallocation entry points.
 */
class MemoryManager {
public:
    /**
     * @brief Initialize memory tracking (stub).
     */
    static void init();

    /**
     * @brief Allocate a block of memory (stub).
     */
    static void* allocate(size_t size);

    /**
     * @brief Deallocate a block of memory (stub).
     */
    static void deallocate(void* ptr);

    /**
     * @brief Current total allocated bytes (stub).
     */
    static size_t totalAllocated();

private:
    static size_t total_allocated_;
};

#endif  // MEMORY_MANAGER_H
