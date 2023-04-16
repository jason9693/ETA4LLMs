# ETA4LLMs
see [calc_flops.py](calc_flops.py) for details

## Notation
- N : Number of Layers
- L : Sequence Length (Average)
- $d_e$ : embedding dim
- $d_H$ : head dim (embedding split by heads)
- $n_H$ : number of heads
- $d_{ff}$ : feed forward layer dimension (= $4d_e$)
- S : Number of Steps

## Total FLOP
$$FLOP_{Model}(n_h, d_h, d_e,L,d_{ff},S,B,N) = SBL*3\{(M(n_H,d_H,d_e,L) + {FF}_{sub}(d_e)) *N + FF_{final}(d_e, d_{ff})\}$$

## Expected Spent Time
$$\text{Expected Spent Time} = \alpha*\frac{FLOP_{Model}}{FLOPS}(\text{seconds})$$