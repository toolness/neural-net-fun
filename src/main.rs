mod value;

use value::Value;

fn main() {
    let a = Value::new_param("a", 2.0);
    let b = Value::new_param("b", -3.0);
    let c = Value::new_param("c", 10.0);
    let e = a * b;
    let d = e + c;
    let f = Value::new_param("f", -2.0);
    let mut loss = (d * f).exp();
    loss.backward();
    println!("{loss} = {} (grad={})", loss.as_f64(), loss.grad());
    println!("{loss:#?}");
}
