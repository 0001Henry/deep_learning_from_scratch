```python
import numpy as np
```

## Create


```python
arr1 = np.array([1.,2.,3.])
arr1
```




    array([1., 2., 3.])




```python
arr2 = arr1.astype(int)
arr2
```




    array([1, 2, 3])




```python
arr3 = np.ones((3,2,1))
arr3
```




    array([[[1.],
            [1.]],
    
           [[1.],
            [1.]],
    
           [[1.],
            [1.]]])




```python
a = np.arange(1,21,2)
a
```




    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])




```python
a = np.random.random((3,2))
a
```




    array([[0.30421515, 0.53292776],
           [0.28661949, 0.55374416],
           [0.54982564, 0.451857  ]])




```python
a = np.random.rand(3,2)
a
```




    array([[0.77216069, 0.40891772],
           [0.92578388, 0.08103115],
           [0.51501061, 0.55870138]])




```python
a = np.random.randint(1,10,(2,3))
a
```




    array([[1, 4, 2],
           [3, 2, 9]])




```python
a = np.random.normal(0,1,(2,3))
a
```




    array([[-1.52450694,  0.81997198,  0.22341405],
           [-1.23589707,  1.75922823, -0.77042462]])




```python
a = np.random.randn(2,3)
a
```




    array([[-1.9286721 , -0.49341625, -0.72808546],
           [ 0.56516771, -0.04973357, -0.47164904]])



## Fancy Index


```python
a = np.arange(1,17).reshape(4,4)
a
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])




```python
print(a[[0,1],[0,2]])
print(a[[0,1,2,3],[3,2,1,0]])
print(a[[1],[1]])
print(a[1][1])
```

    [  1 100]
    [100 100 100 100]
    [6]
    6
    


```python
a[[0,1,2,3],[3,2,1,0]] = 100
a
```




    array([[  1,   2,   3, 100],
           [  5,   6, 100,   8],
           [  9, 100,  11,  12],
           [100,  14,  15,  16]])



## Slice


```python
a = np.arange(1,21).reshape(4,5)
b = a.copy()
print(a)
print(b)
a
```

    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20]])




```python
a[1::2,2::]
```




    array([[ 8,  9, 10],
           [18, 19, 20]])




```python
print(a[2])
print(a[2,:])
```

    [11 12 13 14 15]
    [11 12 13 14 15]
    


```python
print(a[:,2])
print(a[:,1:3])
```

    [ 3  8 13 18]
    [[ 2  3]
     [ 7  8]
     [12 13]
     [17 18]]
    


```python
a[:2] = 100
a
```




    array([[100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100],
           [ 11,  12,  13,  14,  15],
           [ 16,  17,  18,  19,  20]])



## Deform


```python
a = np.arange(1,7).reshape(2,3)
a,a.T
```




    (array([[1, 2, 3],
            [4, 5, 6]]),
     array([[1, 4],
            [2, 5],
            [3, 6]]))




```python
a = np.arange(10)
a_ud = np.flipud(a)
a,a_ud
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
a = np.arange(20).reshape(4,5)
a_ud = np.flipud(a)
a_lr = np.fliplr(a)
print(a)
print(a_ud)
print(a_lr)
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]]
    [[15 16 17 18 19]
     [10 11 12 13 14]
     [ 5  6  7  8  9]
     [ 0  1  2  3  4]]
    [[ 4  3  2  1  0]
     [ 9  8  7  6  5]
     [14 13 12 11 10]
     [19 18 17 16 15]]
    


```python
a = np.array([
    [1,2,3],
    [4,5,6]
])
b = np.array([
    [7,8,9],
    [10,11,12]
])
c = np.concatenate([a, b])
d = np.concatenate([a, b], axis=1)
print(c)
print(d)
```

    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]
    [[ 1  2  3  7  8  9]
     [ 4  5  6 10 11 12]]
    

## Functions


```python
a = np.arange(5)
b = np.arange(5)
c = np.dot(a,b)
a,b,c
```




    (array([0, 1, 2, 3, 4]), array([0, 1, 2, 3, 4]), 30)




```python
a = np.arange(12).reshape(3,4)
b = np.arange(20).reshape(4,5)
c = np.dot(a,b)
a,b,c
```




    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]),
     array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]]),
     array([[ 70,  76,  82,  88,  94],
            [190, 212, 234, 256, 278],
            [310, 348, 386, 424, 462]]))




```python
x = np.arange(-2,3)
x1 = np.abs(x)
print(x)
print(x1)
```

    [-2 -1  0  1  2]
    [2 1 0 1 2]
    


```python
theta = np.arange(3)*np.pi/2.
print(theta)
print(np.sin(theta))
print(np.cos(theta))
print(np.tan(np.pi/4))
```

    [0.         1.57079633 3.14159265]
    [0.0000000e+00 1.0000000e+00 1.2246468e-16]
    [ 1.000000e+00  6.123234e-17 -1.000000e+00]
    0.9999999999999999
    


```python
a = np.random.randn(10000000)
num = np.sum(a<0)
num
```




    5000092




```python
a = np.arange(11)
b = np.flipud(a)
print(a)
print(b)
print(np.any(a==b))
print(np.all(a==b))
```

    [ 0  1  2  3  4  5  6  7  8  9 10]
    [10  9  8  7  6  5  4  3  2  1  0]
    True
    False
    


```python
print(np.where(a>5))
print(np.where(a==np.max(a)))
```

    (array([ 6,  7,  8,  9, 10], dtype=int64),)
    (array([10], dtype=int64),)
    

# torch

![alt text](assets/numpy_exercise/image.png)

```python
import torch
```


```python
x = np.random.randn(2,8,3,4)
y = torch.randn(2,8,3,4)
z = torch.randn(2,8,4,5)
res = torch.matmul(y,z)
res2 = y@z
print(res.shape)
print(torch.all(res==res2))
```

    torch.Size([2, 8, 3, 5])
    tensor(True)
    


```python
a = torch.arange(5)
b = a.type(torch.float32)
b
```




    tensor([0., 1., 2., 3., 4.])




```python
a = np.arange(5)
b = a.astype(float)
b
```




    array([0., 1., 2., 3., 4.])


