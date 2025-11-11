use nalgebra::DMatrix;
use my_library::{get_a_matrix, predict_kdmd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample time series data - similar to the R example
    let data = DMatrix::from_row_slice(3, 10, &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    
    println!("Original data:");
    println!("{:.3}", data);
    
    // Create Koopman matrix
    let kdmd = get_a_matrix(&data, 0.9, None)?;
    
    println!("\\nKoopman matrix:");
    println!("{:.6}", kdmd.as_matrix());
    
    // Predict future time steps
    let prediction = predict_kdmd(&kdmd, &data, 2)?;
    
    println!("\\nPrediction result:");
    println!("{:.3}", prediction);
    
    Ok(())
}
