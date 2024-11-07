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

## Usage

The smoothing parameter for HRV data should be lambda=500. This corresponds to a time-varying FIR high-pass filter with a cut-off frequency of 0.033 Hz

This is according to "Entropy in Heart Rate Dynamics Refleccts How HRV-Biofeedback Training Improves NEurovisceral Complexity during Stress-Cognition Interactions"

While this filter use will make a difference in removing low end power from an FFT, it is not clear that it makes any difference to entropy calculations on short time spans of RR data. These are too short to have any components down in that frequency range.
