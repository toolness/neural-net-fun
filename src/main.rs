mod value;

use value::Value;

fn main() {
    let x = Value::new_param("x", 1.0);
    let sum = x + 3.0.into();
    println!("{sum} = {}", sum.as_f64());
}
