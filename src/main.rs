mod perceptron;
mod plot;
mod value;
mod vec3;

use value::Value;

use macroquad::prelude::*;

use perceptron::{Datapoint, Perceptron};
use plot::Plot;

/// Each point on the plot is scaled by this many screen pixels.
const PLOT_SCALE: f32 = 8.0;

/// Max time in ms between perceptron auto-updates (when auto-updating
/// is enabled).
const MAX_AUTO_UPDATE_TIME: i32 = 250;

/// Amount to increment/decrement auto-update, in ms, when user presses
/// keys to do so.
const AUTO_UPDATE_INCREMENT: i32 = 25;

/// Empty space between left side of screen and text, in pixels.
const LEFT_PADDING: f32 = 10.0;

const HELP_TEXT: &'static str = r#"Help

H - Toggle help
SPACE - Update perceptron
A - Toggle auto-update mode
[ - Decrease auto-update speed
] - Increase auto-update speed
1 - Paint green datapoint (at mouse cursor)
2 - Paint purple datapoint (at mouse cursor)
X - Delete datapoint (at mouse cursor)
C - Clear all datapoints
W - Reset weights to zero
"#;

#[macroquad::main("Perceptron Fun")]
async fn main() {
    run_smoke_test();

    let mut datapoints = vec![
        Datapoint::new((3, 1), 1),
        Datapoint::new((2, -1), 1),
        Datapoint::new((-2, 1), 1),
        Datapoint::new((-1, -3), 0),
        //Datapoint::new((47, -23), 0),
    ];
    let mut perceptron = Perceptron::new(datapoints.clone(), Default::default());

    let plot = Plot::new(PLOT_SCALE);
    let mut auto_update_time = 0; //MAX_AUTO_UPDATE_TIME;
    let mut auto_update = false;
    let mut show_help = false;
    let mut last_frame_time = get_time();
    let mut time_to_auto_update = MAX_AUTO_UPDATE_TIME as f64 / 1000.0;
    let help_lines: Vec<&'static str> = HELP_TEXT.split('\n').collect();

    loop {
        let now = get_time();
        let delta_time = now - last_frame_time;
        last_frame_time = now;

        clear_background(BLACK);

        let mouse_f32 = plot.from_screen_point(mouse_position());
        let mouse = (mouse_f32.0.round() as i32, mouse_f32.1.round() as i32);

        let did_modify_datapoints = if is_key_down(KeyCode::Key1) {
            modify_datapoint(&mut datapoints, mouse, Some(1))
        } else if is_key_down(KeyCode::Key2) {
            modify_datapoint(&mut datapoints, mouse, Some(0))
        } else if is_key_down(KeyCode::X) {
            modify_datapoint(&mut datapoints, mouse, None)
        } else if is_key_pressed(KeyCode::C) {
            datapoints = vec![];
            true
        } else {
            false
        };

        if did_modify_datapoints {
            perceptron = Perceptron::new(datapoints.clone(), perceptron.weights());
        } else if is_key_pressed(KeyCode::W) {
            perceptron = Perceptron::new(datapoints.clone(), Default::default());
        }

        if is_key_pressed(KeyCode::H) {
            show_help = !show_help;
        }

        if is_key_pressed(KeyCode::A) {
            auto_update = !auto_update;
            time_to_auto_update = 0.0;
        }

        if is_key_pressed(KeyCode::LeftBracket) {
            // The reason we store auto-update numbers as integers is so we can use std::cmp
            // functions, as floats aren't fully-ordered.
            auto_update_time = std::cmp::min(
                auto_update_time + AUTO_UPDATE_INCREMENT,
                MAX_AUTO_UPDATE_TIME,
            );
        } else if is_key_pressed(KeyCode::RightBracket) {
            auto_update_time = std::cmp::max(auto_update_time - AUTO_UPDATE_INCREMENT, 0);
        }

        let should_update = if auto_update {
            time_to_auto_update -= delta_time;
            if time_to_auto_update <= 0.0 {
                time_to_auto_update = auto_update_time as f64 / 1000.0;
                true
            } else {
                false
            }
        } else {
            is_key_pressed(KeyCode::Space)
        };

        if should_update {
            perceptron.update();
        }

        plot.draw_axes();
        plot.draw_circle(mouse.0 as f32, mouse.1 as f32, 0.75, DARKGRAY);

        perceptron.draw(&plot);

        let auto_label = if auto_update { "AUTO" } else { "" };
        let conv_label = if perceptron.has_converged() {
            "CONVERGENCE"
        } else {
            ""
        };
        draw_text(
            &format!(
                "Loss: {:0.4?} {auto_label:4} {conv_label:11}",
                perceptron.loss()
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
    datapoints: &mut Vec<Datapoint>,
    point: (i32, i32),
    label: Option<i32>,
) -> bool {
    if let Some(label) = label {
        if let Some(dp) = datapoints.iter_mut().find(|dp| dp.pos == point) {
            if dp.label != label {
                println!("Changing label of {point:?} to {label}.");
                dp.label = label;
                return true;
            }
        } else {
            println!("Adding datapoint at {point:?} with label {label}.");
            datapoints.push(Datapoint::new(point, label));
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
