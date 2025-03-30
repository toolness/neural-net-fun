mod context;
mod value;

use context::{Context, new_param};
use value::Value;

fn main() {
    {
        let x = Value::new_param("x", 1.0);
        let sum = x + 3.0.into();
        println!("{sum} = {}", sum.as_f64());
    }

    {
        let mut ctx = Context::default();
        let x = ctx.param("x".into(), 1.0);
        let y = ctx.param("y".into(), 3.0);
        let sum = ctx.sum(x, y);
        println!("{} = {}", ctx.expr(sum), ctx.compute(sum));
        ctx.set(x, 2.0);
        println!("{} = {}", ctx.expr(sum), ctx.compute(sum));

        let double_sum = ctx.sum(sum, sum);
        println!("{} = {}", ctx.expr(double_sum), ctx.compute(double_sum));
    }

    {
        let mut x = new_param("x", 1.0);
        let sum = x + 3.0.into();
        println!("{sum} = {}", sum.compute());
        x.set(2.0);
        println!("{sum} = {}", sum.compute());
    }
}
