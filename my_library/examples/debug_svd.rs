use nalgebra::DMatrix;
use my_library::{get_a_matrix, predict_kdmd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the exact same data as in R examples
    let data = DMatrix::from_row_slice(3, 10, &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]);
    
    println!("Testing with p=1.0 (full SVD):");
    let kdmd = get_a_matrix(&data, 1.0, None)?;
    
    println!("Koopman matrix:");
    println!("{:.10}", kdmd.as_matrix());
    
    let prediction = predict_kdmd(&kdmd, &data, 3)?;
    let predicted_cols = prediction.columns(data.ncols(), 3);
    println!("Predicted columns:");
    println!("{:.6}", predicted_cols);
    
    println!("Expected: [11, 12, 13], [12, 13, 14], [13, 14, 15]");
    
    Ok(())
}
