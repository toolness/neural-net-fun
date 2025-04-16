use macroquad::prelude::*;

pub struct Plot {
    scale: f32,
}

impl Plot {
    pub fn new(scale: f32) -> Self {
        Plot { scale }
    }

    fn origin_x(&self) -> f32 {
        (screen_width() / 2.0).floor()
    }

    fn origin_y(&self) -> f32 {
        (screen_height() / 2.0).floor()
    }

    fn screen_x(&self, x: f32) -> f32 {
        self.origin_x() + x * self.scale
    }

    fn screen_y(&self, y: f32) -> f32 {
        self.origin_y() + y * -self.scale
    }

    pub fn from_screen_point(&self, (x, y): (f32, f32)) -> (f32, f32) {
        (
            (x - self.origin_x()) / self.scale,
            (y - self.origin_y()) / -self.scale,
        )
    }

    pub fn draw_axes(&self) {
        draw_line(
            0.0,
            self.origin_y(),
            screen_width(),
            self.origin_y(),
            1.0,
            DARKGRAY,
        );
        draw_line(
            self.origin_x(),
            0.0,
            self.origin_x(),
            screen_height(),
            1.0,
            DARKGRAY,
        );
    }

    pub fn _draw_line(&self, x1: f32, y1: f32, x2: f32, y2: f32, color: Color) {
        draw_line(
            self.screen_x(x1),
            self.screen_y(y1),
            self.screen_x(x2),
            self.screen_y(y2),
            1.0,
            color,
        );
    }

    pub fn draw_circle(&self, x: f32, y: f32, r: f32, color: Color) {
        draw_circle(self.screen_x(x), self.screen_y(y), r * self.scale, color);
    }

    pub fn draw_point(&self, x: f32, y: f32, color: Color) {
        draw_circle(self.screen_x(x), self.screen_y(y), 1.0, color);
    }
}
