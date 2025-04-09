use macroquad::prelude::*;

#[derive(Default)]
pub struct Button {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    background: Color,
    text: Option<(&'static str, f32, Color)>,
}

impl Button {
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

    pub fn with_text(mut self, text: &'static str, size: f32, color: Color) -> Self {
        self.text = Some((text, size, color));
        self
    }

    pub fn clicked(&self) -> bool {
        draw_rectangle(self.x, self.y, self.width, self.height, self.background);
        draw_rectangle_lines(self.x, self.y, self.width, self.height, 2.0, WHITE);
        if let Some((text, size, color)) = self.text {
            let metrics = measure_text(text, None, size as u16, 1.0);
            let (center_x, center_y) = (self.x + (self.width / 2.0), self.y + (self.height / 2.0));
            draw_text(
                text,
                center_x - metrics.width / 2.0,
                center_y - metrics.height / 2.0 + metrics.offset_y,
                size,
                color,
            );
        }
        if is_mouse_button_pressed(MouseButton::Left) {
            let (x, y) = mouse_position();
            if x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height {
                return true;
            }
        }
        false
    }
}
