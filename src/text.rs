use std::cell::RefCell;

use macroquad::prelude::*;

thread_local! {
    pub static CUSTOM_FONT: RefCell<Font> = RefCell::new(load_ttf_font_from_bytes(include_bytes!("../JetBrainsMono-Medium.ttf")).unwrap());
}

pub fn draw_custom_text(
    text: &str,
    x: f32,
    y: f32,
    font_size: u16,
    color: Color,
) -> TextDimensions {
    CUSTOM_FONT.with_borrow(|font| {
        draw_text_ex(
            text,
            x,
            y,
            TextParams {
                font: Some(font),
                font_size,
                font_scale: 1.0,
                color,
                ..Default::default()
            },
        )
    })
}
