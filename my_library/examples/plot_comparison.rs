use my_library::{get_a_matrix, predict_kdmd};
use nalgebra::DMatrix;
use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== DYNAMIC MODE DECOMPOSITION VISUALIZATION ===");
    
    // Create the same data matrix as in R walk-through: 3x100 sin waves
    let mut data = DMatrix::zeros(3, 100);
    
    // Generate exactly as R: x <- 1:100, m[1,] <- sin(x*pi/10), etc.
    for i in 1..=100 {
        let x = i as f64;
        data[(0, i-1)] = (x * std::f64::consts::PI / 10.0).sin();
        data[(1, i-1)] = (1.0 + x * std::f64::consts::PI / 10.0).sin();
        data[(2, i-1)] = (2.0 + x * std::f64::consts::PI / 10.0).sin();
    }
    
    println!("📊 Generated 3x100 sin wave data matrix");
    println!("Variables:");
    println!("  Row 1: sin(t*π/10)");
    println!("  Row 2: sin(1 + t*π/10)"); 
    println!("  Row 3: sin(2 + t*π/10)");
    
    println!("\n🔬 Computing Dynamic Mode Decomposition...");
    let kdmd = get_a_matrix(&data, 1.0, None)?;
    
    println!("\n🔮 Predicting 100 future time steps...");
    let extended_data = predict_kdmd(&kdmd, &data, 100)?;
    
    println!("✅ Extended time series:");
    println!("  Original data: t = 1 to 100 (training)");
    println!("  Predictions:   t = 101 to 200 (forecast)");
    
    println!("\n📈 Creating visualization...");
    create_comparison_plot(&extended_data)?;
    
    println!("✅ Plot saved as 'dmd_comparison.png'");
    
    // Show prediction quality at a few points
    println!("\n📊 PREDICTION ACCURACY ANALYSIS:");
    for t in [101, 110, 120, 150, 200] {
        let col_idx = t - 1;
        if col_idx < extended_data.ncols() {
            let actual = [
                (t as f64 * std::f64::consts::PI / 10.0).sin(),
                (1.0 + t as f64 * std::f64::consts::PI / 10.0).sin(),
                (2.0 + t as f64 * std::f64::consts::PI / 10.0).sin()
            ];
            
            println!("t={}:", t);
            for var in 0..3 {
                let predicted = extended_data[(var, col_idx)];
                let error = (predicted - actual[var]).abs();
                let rel_error = if actual[var] != 0.0 { 100.0 * error / actual[var].abs() } else { 0.0 };
                println!("  Var {}: pred={:>7.4}, actual={:>7.4}, error={:>6.2}%", 
                         var+1, predicted, actual[var], rel_error);
            }
        }
    }
    
    Ok(())
}

fn create_comparison_plot(data: &DMatrix<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("dmd_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("DMD Predictions vs Actual Sinusoidal Data", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(1f64..200f64, -1.5f64..1.5f64)?;

    chart
        .configure_mesh()
        .x_desc("Time")
        .y_desc("Value")
        .axis_desc_style(("sans-serif", 25))
        .label_style(("sans-serif", 20))
        .draw()?;

    // Plot Variable 1 - training data
    let var1_training: Vec<(f64, f64)> = (0..100)
        .map(|col| ((col + 1) as f64, data[(0, col)]))
        .collect();
    
    chart
        .draw_series(LineSeries::new(var1_training, &RED))?
        .label("Variable 1 - Training");

    // Plot Variable 1 - predictions
    let var1_predictions: Vec<(f64, f64)> = (100..data.ncols())
        .map(|col| ((col + 1) as f64, data[(0, col)]))
        .collect();
    
    chart
        .draw_series(LineSeries::new(var1_predictions, RED.mix(0.5)))?
        .label("Variable 1 - Predictions");

    // Add vertical line at t=100
    chart
        .draw_series(LineSeries::new(
            vec![(100.5, -1.5), (100.5, 1.5)], 
            &BLACK
        ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    Ok(())
}
