use my_library::get_a_matrix;
use nalgebra::DMatrix;

fn main() {
    let mut data = DMatrix::zeros(3, 100);
    for i in 1..=100 {
        let x = i as f64;
        data[(0, i-1)] = (x * std::f64::consts::PI / 10.0).sin();
        data[(1, i-1)] = (1.0 + x * std::f64::consts::PI / 10.0).sin();
        data[(2, i-1)] = (2.0 + x * std::f64::consts::PI / 10.0).sin();
    }
    
    match get_a_matrix(&data, 1.0, None) {
        Ok(kdmd) => {
            println!("Success! Koopman matrix computed with complex arithmetic");
            let matrix = kdmd.as_matrix();
            println!("Matrix size: {}x{}", matrix.nrows(), matrix.ncols());
        },
        Err(e) => println!("Error: {}", e),
    }
}
