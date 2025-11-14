use my_library::{get_a_matrix, predict_kdmd};
use nalgebra::DMatrix;
use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== DYNAMIC MODE DECOMPOSITION WITH PLOTTING ===");
    println!("Replicating R walk-through with visualization");
    
    // Create the same data matrix as in R walk-through: 3x100 sin waves
    let mut data = DMatrix::zeros(3, 100);
    
    // Generate exactly as R: x <- 1:100, m[1,] <- sin(x*pi/10), etc.
    for i in 1..=100 {
        let x = i as f64;
        data[(0, i-1)] = (x * std::f64::consts::PI / 10.0).sin();
        data[(1, i-1)] = (1.0 + x * std::f64::consts::PI / 10.0).sin();
        data[(2, i-1)] = (2.0 + x * std::f64::consts::PI / 10.0).sin();
    }
    
    println!("\n📊 Generated 3x100 sin wave data matrix");
    println!("Variables:");
    println!("  Row 1: sin(t*π/10)");
    println!("  Row 2: sin(1 + t*π/10)"); 
    println!("  Row 3: sin(2 + t*π/10)");
    
    println!("\n🔬 Computing Dynamic Mode Decomposition...");
    let kdmd = get_a_matrix(&data, 1.0, None)?;
    let koopman_matrix = kdmd.as_matrix();
    
    println!("✅ Koopman Operator Matrix (A):");
    for i in 0..koopman_matrix.nrows() {
        print!("  [");
        for j in 0..koopman_matrix.ncols() {
            print!("{:>10.6}", koopman_matrix[(i, j)]);
            if j < koopman_matrix.ncols() - 1 { print!("  "); }
        }
        println!(" ]");
    }
    
    println!("\n🔮 Predicting 100 future time steps...");
    println!("Using equation: x(t+1) = A * x(t)");
    let prediction = predict_kdmd(&kdmd, &data, 100)?;
    
    println!("✅ Extended time series:");
    println!("  Original data: t = 1 to 100 (training)");
    println!("  Predictions:   t = 101 to 200 (forecast)");
    
    println!("\n📈 Creating visualization...");
    create_plot(&prediction)?;
    
    println!("✅ Plot saved as 'dmd_prediction.png'");
    println!("\n🎯 PLOT DESCRIPTION:");
    println!("The plot replicates the R walk-through visualization showing:");
    println!("  • Red line:   Variable 1 = sin(t*π/10)");
    println!("  • Blue line:  Variable 2 = sin(1 + t*π/10)");
    println!("  • Green line: Variable 3 = sin(2 + t*π/10)");
    println!("  • Vertical black line: Separates training (left) from predictions (right)");
    println!("  • Time range: 1-200 (100 original + 100 predicted time points)");
    
    // Show prediction quality at the boundary
    println!("\n📊 PREDICTION ACCURACY AT BOUNDARY:");
    let t = 101.0;
    let actual_101 = [
        (t * std::f64::consts::PI / 10.0).sin(),
        (1.0 + t * std::f64::consts::PI / 10.0).sin(),
        (2.0 + t * std::f64::consts::PI / 10.0).sin()
    ];
    
    for i in 0..3 {
        let predicted = prediction[(i, 100)]; // Column 100 is t=101
        let actual = actual_101[i];
        let error = (predicted - actual).abs();
        let rel_error = if actual != 0.0 { 100.0 * error / actual.abs() } else { f64::NAN };
        println!("  Variable {}: predicted = {:>8.4}, actual = {:>8.4}, error = {:>6.2}%", 
                 i+1, predicted, actual, rel_error);
    }
    
    Ok(())
}

fn create_plot(data: &DMatrix<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("dmd_prediction.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Dynamic Mode Decomposition: Original Data + Predictions", ("sans-serif", 36))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(1f64..200f64, -2.5f64..2.5f64)?;

    chart
        .configure_mesh()
        .x_desc("Time")
        .y_desc("Value")
        .axis_desc_style(("sans-serif", 24))
        .label_style(("sans-serif", 18))
        .draw()?;

    // Define colors and labels matching R plot style
    let colors = [&RED, &BLUE, &GREEN];
    let labels = ["sin(t*π/10)", "sin(1 + t*π/10)", "sin(2 + t*π/10)"];
    
    // Plot each variable with legend
    for row in 0..3 {
        let points: Vec<(f64, f64)> = (0..data.ncols())
            .map(|col| ((col + 1) as f64, data[(row, col)]))
            .collect();
        
        chart
            .draw_series(LineSeries::new(points, colors[row].stroke_width(2)))?
            .label(labels[row])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], colors[row].stroke_width(3)));
    }
    
    // Add a vertical line at t=100.5 to separate original data from predictions
    chart
        .draw_series(LineSeries::new(
            vec![(100.5, -2.5), (100.5, 2.5)], 
            BLACK.mix(0.7).stroke_width(2)
        ))?
        .label("Training/Prediction Boundary")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 15, y)], BLACK.stroke_width(3)));
    
    // Add text annotations
    chart.draw_series(std::iter::once(Text::new("Training Data\n(t = 1-100)", (50.0, 2.2), ("sans-serif", 20))))?;
    chart.draw_series(std::iter::once(Text::new("DMD Predictions\n(t = 101-200)", (150.0, 2.2), ("sans-serif", 20))))?;
    
    // Configure legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;
    println!("✅ High-quality plot rendered: 1200x800 pixels");
    Ok(())
}
