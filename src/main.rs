mod classifier_2d;
mod engine;
mod plot;
mod value;

use value::Value;

use macroquad::prelude::*;

use classifier_2d::{Classifier2D, Datapoint2D, Label2D, Weights2D};
use plot::Plot;

/// Each point on the plot is scaled by this many screen pixels.
const PLOT_SCALE: f32 = 8.0;

/// Empty space between left side of screen and text, in pixels.
const LEFT_PADDING: f32 = 10.0;

/// Minimum learning speed value, as an integer.
const MIN_LEARN_SPEED: i32 = 1;

/// The learning speed is multiplied by this scaling factor to
/// arrive at the actual learning rate.
const LEARN_SCALE: f64 = 0.05;

/// Maximum number of hidden layers in the multi-layer perceptron.
const MAX_HIDDEN_LAYERS: usize = 3;

/// Number of neuron in each hidden layer of the multi-layer perceptron.
///
/// This is mentioned in `HELP_TEXT` below, so if you change it here,
/// change it there too.
const NEURONS_PER_LAYER: usize = 16;

const HELP_TEXT: &'static str = r#"Help

H - Toggle help
[ - Decrease updates per frame
] - Increase updates per frame
, - Decrease learning rate
. - Increase learning rate
1 - Paint green datapoint (at mouse cursor)
2 - Paint purple datapoint (at mouse cursor)
X - Delete datapoint (at mouse cursor)
L - Cycle number of hidden layers (16 neurons per layer)
C - Clear all datapoints
W - Reset weights
S - Toggle point mesh shading
"#;

#[macroquad::main("Perceptron Fun")]
async fn main() {
    run_smoke_test();

    let mut datapoints = vec![
        Datapoint2D::new((-10, 9), Label2D::Green),
        Datapoint2D::new((8, 8), Label2D::Green),
        Datapoint2D::new((-5, -5), Label2D::Purple),
        Datapoint2D::new((9, -10), Label2D::Purple),
    ];
    let mut num_hidden_layers = 0;
    let mut perceptron = make_perceptron(&datapoints, num_hidden_layers);

    let plot = Plot::new(PLOT_SCALE);
    let mut updates_per_frame = 1;
    let mut enable_shading = false;
    let mut show_help = false;
    let mut learning_speed = 2;
    let help_lines: Vec<&'static str> = HELP_TEXT.split('\n').collect();

    loop {
        clear_background(BLACK);

        let mouse_f32 = plot.from_screen_point(mouse_position());
        let mouse = (mouse_f32.0.round() as i32, mouse_f32.1.round() as i32);

        let did_modify_datapoints = if is_key_down(KeyCode::Key1) {
            modify_datapoint(&mut datapoints, mouse, Some(Label2D::Green))
        } else if is_key_down(KeyCode::Key2) {
            modify_datapoint(&mut datapoints, mouse, Some(Label2D::Purple))
        } else if is_key_down(KeyCode::X) {
            modify_datapoint(&mut datapoints, mouse, None)
        } else if is_key_pressed(KeyCode::C) {
            datapoints = vec![];
            true
        } else {
            false
        };

        if did_modify_datapoints {
            perceptron = Classifier2D::new(datapoints.clone(), perceptron.weights());
        } else if is_key_pressed(KeyCode::W) {
            perceptron = make_perceptron(&datapoints, num_hidden_layers);
        } else if is_key_pressed(KeyCode::L) {
            num_hidden_layers = (num_hidden_layers + 1) % MAX_HIDDEN_LAYERS;
            perceptron = make_perceptron(&datapoints, num_hidden_layers);
        }

        if is_key_pressed(KeyCode::H) {
            show_help = !show_help;
        }

        if is_key_pressed(KeyCode::S) {
            enable_shading = !enable_shading;
        }

        if is_key_pressed(KeyCode::LeftBracket) {
            updates_per_frame = std::cmp::max(updates_per_frame - 1, 0);
        } else if is_key_pressed(KeyCode::RightBracket) {
            updates_per_frame += 1;
        }

        if is_key_pressed(KeyCode::Comma) {
            // The reason we store learning rate-related numbers as integers is so we can use std::cmp
            // functions, as floats aren't fully-ordered.
            learning_speed = std::cmp::max(learning_speed - 1, MIN_LEARN_SPEED);
        } else if is_key_pressed(KeyCode::Period) {
            learning_speed += 1;
        }

        let learning_rate = learning_speed as f64 * LEARN_SCALE;

        for _ in 0..updates_per_frame {
            perceptron.update(learning_rate);
        }

        plot.draw_axes();
        plot.draw_circle(mouse.0 as f32, mouse.1 as f32, 0.75, DARKGRAY);

        perceptron.draw(&plot, enable_shading);

        draw_text(
            &format!(
                "Loss: {:0.4?} Acc: {}% LR: {:0.3} Spd: {updates_per_frame} HL: {num_hidden_layers} ({} params)",
                perceptron.loss(),
                (perceptron.accuracy() * 100.0).floor(),
                learning_rate,
                perceptron.num_params()
            ),
            LEFT_PADDING,
            screen_height() - 30.0,
            30.0,
            WHITE,
        );

        if show_help {
            for (index, line) in help_lines.iter().enumerate() {
                draw_text(
                    line,
                    LEFT_PADDING,
                    30.0 + (index as f32 * 30.0),
                    30.0,
                    WHITE,
                );
            }
        } else {
            draw_text("Press H for help.", LEFT_PADDING, 30.0, 30.0, WHITE);
        }

        next_frame().await;

        if is_key_pressed(KeyCode::Escape) {
            break;
        }
    }
}

/// Modifies the datapoint with the given point.
///
/// If the label is none, the datapoint is removed (if it exists).
///
/// Otherwise, the datapoint is modified to have the given label (or if it's not in
/// datapoints, it's added).
///
/// Returns whether the datapoints were changed.
fn modify_datapoint(
    datapoints: &mut Vec<Datapoint2D>,
    point: (i32, i32),
    label: Option<Label2D>,
) -> bool {
    if let Some(label) = label {
        if let Some(dp) = datapoints.iter_mut().find(|dp| dp.pos == point) {
            if dp.label != label {
                println!("Changing label of {point:?} to {label:?}.");
                dp.label = label;
                return true;
            }
        } else {
            println!("Adding datapoint at {point:?} with label {label:?}.");
            datapoints.push(Datapoint2D::new(point, label));
            return true;
        }
    } else {
        if let Some(pos) = datapoints.iter().position(|dp| dp.pos == point) {
            println!("Removing datapoint at {point:?}.");
            datapoints.remove(pos);
            return true;
        }
    }
    false
}

fn make_perceptron(datapoints: &Vec<Datapoint2D>, num_hidden_layers: usize) -> Classifier2D {
    Classifier2D::new(
        datapoints.clone(),
        Weights2D::new((0..num_hidden_layers).map(|_| NEURONS_PER_LAYER).collect()),
    )
}

fn run_smoke_test() {
    let a = Value::new_param("a", 2.0);
    let b = Value::new_param("b", -3.0);
    let c = Value::new_param("c", 10.0);
    let e = a * b;
    let d = e + c;
    let f = Value::new_param("f", -2.0);
    let mut loss = (d * f).exp().pow(2.0);
    loss.backward();
    //println!("{loss} = {} (grad={})", loss.as_f64(), loss.grad());
    //println!("{loss:#?}");
}
