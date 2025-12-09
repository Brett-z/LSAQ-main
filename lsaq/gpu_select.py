import GPUtil

def gpu_select():

    gpus = GPUtil.getGPUs()
    free_memory = []

    for gpu in gpus:
        free_memory.append(gpu.memoryFree)

    memory_sort = sorted(range(len(free_memory)), key=lambda i: free_memory[i])

    gpu_id = memory_sort[-1]
    gpu_memory = free_memory[memory_sort[-1]]

    print(f'gpu_id:{gpu_id}; gpu_memory:{gpu_memory}')

    return gpu_id, gpu_memory