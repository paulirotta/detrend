# Detrending HRV data

This is a type of low pass filter to prepare data for further analysis. See "An advanced detrending method with application to HRV analysis", Mika P. Tarvainen, Perttu O. Ranta-aho, and Pasi A. Karjalainen

```rust

To test the implementation, you should try it with some real HRV data and compare the results with the MATLAB version.

## Matlab reference code

Test this code against the following using real HRV data to confirm

```matlab
T = length(z);
lambda = 10;
I = speye(T);
D2 = spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
z_stat = (I - inv(I + lambda^2 * D2' * D2)) * z;
```

## Revised and possibly fixed version

```matlab
T = length(z);
lambda = 10;
I = speye(T);
D2 = spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
A = I + lambda^2 * D2' * D2;
s = A \ z; % Solve for the smooth trend component
z_detrended = z - s; % Compute the detrended signal
```
