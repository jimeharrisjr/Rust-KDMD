use nalgebra::DMatrix;
use my_library::{get_a_matrix, KdmdError};

fn main() -> Result<(), KdmdError> {
    // Create the same data matrix as in R walk-through: 3x100 sin waves
    let mut data = DMatrix::zeros(3, 100);
    
    // Generate exactly as R: x <- 1:100, m[1,] <- sin(x*pi/10), etc.
    for i in 1..=100 {
        let x = i as f64;
        data[(0, i-1)] = (x * std::f64::consts::PI / 10.0).sin();
        data[(1, i-1)] = (1.0 + x * std::f64::consts::PI / 10.0).sin();
        data[(2, i-1)] = (2.0 + x * std::f64::consts::PI / 10.0).sin();
    }
    
    println!("Data generation debug - first 5 columns:");
    for row in 0..3 {
        print!("Row {}: ", row + 1);
        for col in 0..5 {
            print!("{:.7} ", data[(row, col)]);
        }
        println!();
    }
    
    // Call get_a_matrix with same parameters as R (p=1.0, comp=None)
    let result = get_a_matrix(&data, 1.0, None)?;
    println!("\nRust Koopman matrix:");
    println!("{}", result.as_matrix());
    
    println!("\nExpected R output:");
    println!("-0.4567364   1.674091  -1.2093754");  
    println!("-2.1796212   3.107948  -1.8123870");
    println!("-1.8939332   1.679359  -0.7444591");
    
    Ok(())
}
