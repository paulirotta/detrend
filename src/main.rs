use ndarray::{Array1, Array2};
use ndarray_inverse::*; // Import the Inverse trait from ndarray-inverse

fn construct_d2(t: usize) -> Array2<f64> {
    let mut d2 = Array2::<f64>::zeros((t - 2, t));
    for i in 0..(t - 2) {
        d2[(i, i)] = 1.0;
        d2[(i, i + 1)] = -2.0;
        d2[(i, i + 2)] = 1.0;
    }
    d2
}

fn detrend_signal(z: &[f64], lambda: f64) -> Vec<f64> {
    let t = z.len();
    if t < 3 {
        return z.to_vec();
    }

    // Convert input slice to Array1
    let z_array = Array1::from(z.to_vec());

    // Identity matrix I
    let i = Array2::<f64>::eye(t);

    // Construct the second-order difference matrix D2
    let d2 = construct_d2(t);

    // Compute A = I + λ² D2ᵗ D2
    let lambda_squared = lambda * lambda;
    let d2t_d2 = d2.t().dot(&d2);
    let a = &i + &(lambda_squared * d2t_d2);

    // Invert the matrix A using ndarray-inverse
    let a_inv = a.inv().unwrap();

    // Solve the linear system A s = z
    let s = a_inv.dot(&z_array);

    // Compute the detrended signal: z_detrended = z_array - s
    let z_detrended = z_array - &s;

    // Return the detrended signal as Vec<f64>
    z_detrended.to_vec()
}

fn main() {
    // Corrected data without the outliers
    let z = vec![
        0.256, 0.357, 0.533, 0.372, 0.712, 0.744, 0.761, 0.525, 0.915, 0.725, 0.739, 0.764, 0.754,
        0.718, 0.707, 0.697, 0.699, 0.718, 0.931, 0.829, 0.814,
    ];
    let lambda = 10.0;
    let detrended_signal = detrend_signal(&z, lambda);
    println!("Detrended Signal: {:?}", detrended_signal);
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    #[test]
    fn test_detrend_signal() {
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
        let detrended_signal = detrend_signal(&z, lambda);

        // Use approx_eq! macro from float-cmp to test approximate equality
        for (computed, expected) in detrended_signal.iter().zip(expected_detrended.iter()) {
            assert!(
                approx_eq!(f64, *computed, *expected, epsilon = 1e-6),
                "Values are not approximately equal: computed = {}, expected = {}",
                computed,
                expected
            );
        }
    }
}
