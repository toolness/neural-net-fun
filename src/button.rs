use macroquad::prelude::*;

use crate::text::{CUSTOM_FONT, draw_custom_text};

#[derive(Default)]
pub struct Button<'a> {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    background: Color,
    text: Option<(&'a str, u16, Color)>,
}

impl<'a> Button<'a> {
    pub fn at(rect: Rect) -> Self {
        Button {
            x: rect.x,
            y: rect.y,
            width: rect.w,
            height: rect.h,
            ..Default::default()
        }
    }

    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn with_text(mut self, text: &'a str, size: u16, color: Color) -> Self {
        self.text = Some((text, size, color));
        self
    }

    fn draw_background(&self) {
        draw_rectangle(self.x, self.y, self.width, self.height, self.background);
    }

    fn draw_foreground(&self) {
        draw_rectangle_lines(self.x, self.y, self.width, self.height, 2.0, WHITE);
        if let Some((text, size, color)) = self.text {
            let metrics =
                CUSTOM_FONT.with_borrow(|font| measure_text(text, Some(font), size as u16, 1.0));
            let (center_x, center_y) = (self.x + (self.width / 2.0), self.y + (self.height / 2.0));
            draw_custom_text(
                text,
                center_x - metrics.width / 2.0,
                center_y - metrics.height / 2.0 + metrics.offset_y,
                size,
                color,
            );
        }
    }

    fn is_mouse_in_bounds(&self) -> bool {
        let (x, y) = mouse_position();
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }

    pub fn slider_value(&self, min: f32, max: f32, step: f32, current: f32, color: Color) -> f32 {
        self.draw_background();
        let value = if is_mouse_button_down(MouseButton::Left) && self.is_mouse_in_bounds() {
            let x = mouse_position().0;
            convert_percentage_to_slider_value((x - self.x) / self.width, min, max, step)
        } else {
            current
        };

        // Draw slider bar.
        let value_pct = (value - min) / (max - min);
        draw_rectangle(self.x, self.y, self.width * value_pct, self.height, color);

        self.draw_foreground();
        value
    }

    pub fn clicked(&self) -> bool {
        self.draw_background();
        self.draw_foreground();
        is_mouse_button_pressed(MouseButton::Left) && self.is_mouse_in_bounds()
    }
}

/// This function was written by Claude.
///
/// `percentage` should be between 0 and 1, representing position on slider.
fn convert_percentage_to_slider_value(percentage: f32, min: f32, max: f32, step: f32) -> f32 {
    // Ensure mouse_x is clamped between 0 and 1
    let clamped_x = percentage.clamp(0.0, 1.0);

    // Calculate the raw value in the target range
    let raw_value = min + clamped_x * (max - min);

    // Calculate how many steps this represents
    let steps = ((raw_value - min) / step).round();

    // Convert back to the actual value aligned to steps
    let result = min + steps * step;

    // Ensure the result doesn't exceed max due to floating point precision
    result.min(max)
}
