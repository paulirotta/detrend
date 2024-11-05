use nalgebra::{DMatrix, DVector};

/// Constructs the second-order difference matrix D2.
/// This matrix is used to compute the detrended signal.
///
/// # Arguments
///
/// * `t` - The length of the input signal.
///
/// # Returns
///
/// A `DMatrix<f64>` representing the D2 matrix of size (t-2) x t.
fn construct_d2(t: usize) -> DMatrix<f64> {
    let mut d2 = DMatrix::<f64>::zeros(t - 2, t);
    for i in 0..(t - 2) {
        d2[(i, i)] = 1.0;
        d2[(i, i + 1)] = -2.0;
        d2[(i, i + 2)] = 1.0;
    }
    d2
}

/// Detrends the input signal `z` using the specified `lambda`.
///
/// # Arguments
///
/// * `z` - A slice of `f64` representing the input signal.
/// * `lambda` - A smoothing parameter.
///
/// # Returns
///
/// A `Result<Vec<f64>, String>` containing the detrended signal or an error message.
fn detrend_signal(z: &[f64], lambda: f64) -> Result<Vec<f64>, String> {
    let t = z.len();
    if t < 3 {
        return Ok(z.to_vec());
    }

    // Convert input slice to DVector
    let z_vector = DVector::from_column_slice(z);

    // Identity matrix I of size t x t
    let i = DMatrix::<f64>::identity(t, t);

    // Construct the second-order difference matrix D2
    let d2 = construct_d2(t);

    // Compute A = I + λ² * D2ᵗ * D2
    let lambda_squared = lambda * lambda;
    let d2t_d2 = d2.transpose() * &d2; // D2ᵗ * D2
    let a = &i + d2t_d2 * lambda_squared; // I + λ² * D2ᵗ * D2

    // Attempt to invert the matrix A
    let a_inv = a
        .try_inverse()
        .ok_or_else(|| "Failed to invert matrix A".to_string())?;

    // Solve the linear system A * s = z by multiplying A_inv with z
    let s = a_inv * z_vector.clone();

    // Compute the detrended signal: z_detrended = z - s
    let z_detrended = &z_vector - &s;

    // Return the detrended signal as Vec<f64>
    Ok(z_detrended.data.as_vec().to_vec())
}

fn main() {
    // Corrected data without the outliers
    let z = vec![
        0.256, 0.357, 0.533, 0.372, 0.712, 0.744, 0.761, 0.525, 0.915, 0.725, 0.739, 0.764, 0.754,
        0.718, 0.707, 0.697, 0.699, 0.718, 0.931, 0.829, 0.814,
    ];
    let lambda = 10.0;

    match detrend_signal(&z, lambda) {
        Ok(detrended_signal) => println!("Detrended Signal: {:?}", detrended_signal),
        Err(e) => eprintln!("Error detrending signal: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_detrend_signal_success() {
        // Prepare the input data
        let z = vec![
            0.256, 0.357, 0.533, 0.372, 0.712, 0.744, 0.761, 0.525, 0.915, 0.725, 0.739, 0.764,
            0.754, 0.718, 0.707, 0.697, 0.699, 0.718, 0.931, 0.829, 0.814,
        ];
        let lambda = 10.0;

        // Expected detrended signal values
        let expected_detrended = vec![
            -0.10157990356948082,
            -0.05846210744017177,
            0.06067148772483172,
            -0.1545786979643745,
            0.13478104072505737,
            0.12229019600561897,
            0.10214044968255054,
            -0.16369941839896396,
            0.20271796710943413,
            -0.004023024259906305,
            -0.001365202645730479,
            0.016288852055814762,
            0.0015502119750396837,
            -0.038132939278303524,
            -0.053327920214212954,
            -0.06922071994990375,
            -0.07546404840044962,
            -0.06701840828142525,
            0.13391033817559939,
            0.01978651282186772,
            -0.00726466587313257,
        ];

        // Call the detrend_signal function
        let detrended_signal = detrend_signal(&z, lambda).expect("Should succeed");

        // Use approx_eq! macro from float-cmp to test approximate equality
        for (i, (computed, expected_val)) in detrended_signal
            .iter()
            .zip(expected_detrended.iter())
            .enumerate()
        {
            assert!(
                approx_eq!(f64, *computed, *expected_val, epsilon = 1e-6),
                "Value at index {}: computed = {}, expected = {}",
                i,
                computed,
                expected_val
            );
        }
    }

    #[test]
    fn test_detrend_signal_short_input() {
        // Test with input length less than 3
        let z = vec![0.5, 0.6];
        let lambda = 5.0;
        let detrended_signal = detrend_signal(&z, lambda).expect("Should return original signal");
        assert_eq!(detrended_signal, z);
    }

    #[test]
    fn test_detrend_signal_inversion_failure() {
        // It's challenging to create a scenario where A is non-invertible with nalgebra and the current construct_d2.
        // Typically, D2ᵗ * D2 adds regularization, making A invertible.
        // However, for completeness, you can mock or simulate an inversion failure if needed.
        // Currently, we'll skip this test.
    }
}
