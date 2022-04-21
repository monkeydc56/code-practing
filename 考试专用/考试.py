def quicksort(arr, low, high):
    i = low-1
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] <= pivot:
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
    arr[high], arr[i+1] = arr[i+1], arr[high]
    return i+1

def startsort(arr, low, high):
    if low < high:
        cut = quicksort(arr, low, high)
        startsort(arr, low, cut-1)
        startsort(arr, cut+1, high)

arr = [1,1,1,1,1,1]
length = len(arr)
startsort(arr, 0, length-1)
print(arr)
