/// Um... on my Android phone, at least, the dpi_scale()
/// is always set to 1.0 even though it's clearly a Retina-style
/// display, so I'm just hard-coding a 2x zoom for now.
#[cfg(target_os = "android")]
const ZOOM_FACTOR: f32 = 2.0;

#[cfg(not(target_os = "android"))]
const ZOOM_FACTOR: f32 = 1.0;

pub fn px(value: f32) -> f32 {
    ZOOM_FACTOR * value
}

pub fn font_scale() -> f32 {
    ZOOM_FACTOR
}
