use nalgebra::DMatrix;
use my_library::{Kdmd, KdmdError};

fn main() -> Result<(), KdmdError> {
    // Test data from R walk-through
    let x = DMatrix::from_row_slice(3, 3, &[
        1.0, 4.0, 7.0,
        2.0, 5.0, 8.0, 
        3.0, 6.0, 9.0
    ]);
    
    let y = DMatrix::from_row_slice(3, 3, &[
        2.0, 5.0, 8.0,
        3.0, 6.0, 9.0,
        4.0, 7.0, 10.0
    ]);
    
    println!("X:\n{}", x);
    println!("Y:\n{}", y);
    
    let result = Kdmd::get_a_matrix(&x, &y)?;
    println!("Koopman matrix:\n{}", result.matrix());
    
    Ok(())
}
