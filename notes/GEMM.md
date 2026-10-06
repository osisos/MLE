# GEMM

代码与运行入口见 [GEMM kernel 示例](../keneral_demo/gemm/README.md)。

## Layout 的复合

Layout A: tile -> reg

```python
Alayout = (2, 8, 4, 2) : (2@reg, 4@laneid, 1@laneid, 1@reg)
```

Layout B: element(x, y) -> tile 

```python
shape  = (128, 128)
Blayout = (8, 16, 16, 8) : (1@m_tile, 0, 1@n_tile, 0)
```

Layout C: element(m_tile, n_tile) -> warp id

```python
CLaytout = (8@m_tile, 4@n_tile, 4@n_tile): (0, 0, 1@warp_id) 
```



如何将两个组合起来，计算出 Layout B x Layout A 的最终复合Layout结果呢

```python
```





```python
C_tile = T.alloc_local(
    (8, 16, 16, 8),
    "float32",
    layout=C_tile_layout,
)

# 先选 MMA tile，再选 tile 内的元素
C_tile[m_tile_idx, n_tile_idx, mi, ni]
```

[8@m_tile, 4@n_tile, 4@n_tile, 2, 8, 4, 2] = [16, 4, 1@warpid, 2, 4@lane_id, 1@lande_id, 1]

[r, c] 到 [m_tile, n_tile]的映射

[128, 128] -> [8, 16]
[128, 128]:[8, 16, 16, 8] [1@m_tile,0,1@n_tile,0]
$$

$$
