package disc

import (
	"fmt"
	"math"
	"sort"
)

func main() {
	var n = []int{9, 1, 33, 21, 78, 42, 4}
	fmt.Println(linearSort(n, 78))
	fmt.Println(binarySearch(n, 1))
	fmt.Println(BubbleSort(n))
	fmt.Println(InsertSort(n))
}

//ПОИСК

// Линейный поиск O(n)
func linearSort(arr []int, s int) int {
	for i, v := range arr {
		if s == v {
			return i
		}
	}
	return -1
}

// Бинарный поиск  O(log n)
func binarySearch(arr []int, s int) int {
	sort.Ints(arr)
	var leftPointer = 0
	var rightPointer = len(arr) - 1
	for leftPointer <= rightPointer {
		var midPointer = int((leftPointer + rightPointer) / 2)
		var midValue = arr[midPointer]
		if midValue == s {
			return midPointer
		} else if midValue < s {
			leftPointer = midPointer + 1
		} else {
			rightPointer = midPointer - 1
		}
	}
	return -1
}

// Прыжковый поиск O(√n)
func jumpSearch(arr []int, s int) int {
	var blockSize = int(math.Sqrt(float64(len(arr))))
	var i = 0
	for {
		if arr[i] >= s {
			break
		}

		if i > len(arr) {
			break
		}
		i = i + blockSize
	}
	for j := i; j > 0; j-- {
		if arr[j] == s {
			return j
		}
	}
	return -1
}

//СОРТИРОВКА

// сортировка пузырьком
func BubbleSort(arr []int) []int {
	for i := 0; i < len(arr)-1; i++ {
		for j := 0; j < len(arr)-i-1; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
			}
		}
	}
	return arr
}

// сортировка вставками
func InsertSort(arr []int) []int {
	lenght := len(arr)
	for i := 1; i < lenght; i++ {
		for j := i; j >= 1 && arr[j] < arr[j-1]; j-- {
			arr[j], arr[j-1] = arr[j-1], arr[j]
		}
	}
	return arr
}

//Сортировка выбором O(n^2)

func SelectionSort(arr []int) []int {
	for i := 1; i < len(arr)-1; i++ {
		j := i + 1
		minIndex := i

		if j < len(arr) {
			if arr[j] < arr[minIndex] {
				minIndex = j
			}
			j++
		}

		if minIndex != i {
			temp := arr[i]
			arr[i] = arr[minIndex]
			arr[minIndex] = temp
		}
		i++
	}
	return arr
}

//Сортировка подсчётом O(n + k)

func CountingSort(arr []int) []int {
	var max = arr[0]
	for i := 1; i < len(arr); i++ {
		if arr[i] > max {
			max = arr[i]
		}
	}
	var indices = make([]int, max+1)
	for j := 0; j < len(arr); j++ {
		indices[arr[j]]++
	}
	for k := 1; k < len(indices); k++ {
		indices[k] = indices[k] + indices[k-1]
	}
	result := make([]int, len(arr))
	for m := 0; m < len(arr); m++ {
		result[indices[arr[m]]-1] = arr[m]
		indices[arr[m]]--
	}
	return result
}

//Переработать //Сортировка слиянием O(n log n)

// func merge(fp []int, sp []int) []int {
// 	arr := make([]int, len(fp)+len(sp))
// 	fpIndex := 0
// 	spIndex := 0
// 	arrIndex := 0

// 	for ; fpIndex < len(fp) && spIndex < len(sp); arrIndex++ {
// 		if fp[fpIndex] < spIndex {
// 			arr[arrIndex] = fp[fpIndex]
// 			fpIndex++
// 		} else if sp[spIndex] < fp[fpIndex] {
// 			arr[arrIndex] = sp[spIndex]
// 			spIndex++
// 		}
// 	}
// 	for ; fpIndex < len(fp); arrIndex++ {
// 		arr[arrIndex] = fp[fpIndex]
// 		fpIndex++
// 	}

// 	for ; spIndex < len(sp); arrIndex++ {
// 		arr[arrIndex] = sp[spIndex]
// 		spIndex++
// 	}
// 	return arr
// }

// func mergeSort(arr []int) []int {
// 	if len(arr) == 1 {
// 		return arr
// 	}
// 	fp := mergeSort(arr[0 : len(arr)/2])
// 	sp := mergeSort(arr[len(arr)/2:])
// 	return merge(fp, sp)
// }

// //////////
