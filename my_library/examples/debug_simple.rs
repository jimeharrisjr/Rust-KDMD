use nalgebra::DMatrix;
use my_library::get_a_matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Very simple test case
    let data = DMatrix::from_row_slice(2, 4, &[
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
    ]);
    
    println!("Testing with simple 2x4 data");
    println!("Data: {:.3}", data);
    
    match get_a_matrix(&data, 1.0, None) {
        Ok(kdmd) => {
            println!("Success! Koopman matrix:");
            println!("{:.6}", kdmd.as_matrix());
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
    
    Ok(())
}
